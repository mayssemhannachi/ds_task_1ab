import os
import json
import logging
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Product Image Dataset with Data Augmentation
class ProductImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        self.logger = logging.getLogger(__name__)
        
        # Additional augmentation for training
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Apply different transforms for training and validation
            if self.is_training:
                image = self.train_transform(image)
            else:
                if self.transform:
                    image = self.transform(image)
            
            label = self.labels[idx]
            return image, label
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            # Return a black image in case of error
            if self.transform:
                return torch.zeros((3, 224, 224)), self.labels[idx]
            return Image.new('RGB', (224, 224), 'black'), self.labels[idx]

# CNN Model
class ProductCNN(nn.Module):
    def __init__(self, num_classes):
        super(ProductCNN, self).__init__()
        
        # Use ResNet18 as base model
        self.base_model = models.resnet18(pretrained=True)
        
        # Freeze early layers
        for param in list(self.base_model.parameters())[:-4]:  # Freeze all except last block
            param.requires_grad = False
        
        # Replace the final layers with custom classifier
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 256),  # Adjusted to 256
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),  # Adjusted to 128
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)  # Ensure num_classes matches the checkpoint
        )
        
        # Image preprocessing for inference
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x):
        return self.base_model(x)
    
    def predict(self, image_path: str) -> Tuple[str, float]:
        """
        Predict the class of a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (predicted_stock_code, confidence)
        """
        self.eval()
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self(image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
            # Convert predicted index to stock code
            predicted_idx = predicted_idx.item()
            confidence = confidence.item()
            
            # Load class mapping
            with open('class_mapping.json', 'r') as f:
                print ()
                class_mapping = json.load(f)
            
            # Convert the mapping to get stock code from index
            idx_to_stock = {v: k for k, v in class_mapping.items()}
            predicted_stock_code = idx_to_stock.get(predicted_idx, f"Unknown class {predicted_idx}")
            
            logger.info(f"Predicted index: {predicted_idx}, Confidence: {confidence}")
            
            return predicted_stock_code, confidence
            
        except Exception as e:
            logger.error(f"Error predicting image {image_path}: {str(e)}")
            return None, 0.0

# Training function with early stopping
def train_model(model, train_loader, valid_loader, num_epochs=50, learning_rate=0.001):
    """
    Train the CNN model with improved training process and overfitting detection
    
    Args:
        model: The CNN model
        train_loader: Training data loader
        valid_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
    """
    logger.info(f"Starting training on device: {model.device}")
    logger.info(f"Training set size: {len(train_loader.dataset)}")
    logger.info(f"Validation set size: {len(valid_loader.dataset)}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_valid_loss = float('inf')
    best_accuracy = 0.0
    patience = 10
    patience_counter = 0
    
    # For overfitting detection
    overfitting_threshold = 0.2  # Max allowed gap between train and val accuracy
    min_epochs = 10  # Minimum epochs before checking overfitting
    consecutive_overfitting = 0
    max_overfitting_epochs = 3
    
    train_history = {'loss': [], 'acc': []}
    valid_history = {'loss': [], 'acc': []}
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch [{epoch+1}/{num_epochs}]")
        logger.info("-" * 50)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_count = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(model.device)
            labels = labels.to(model.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            batch_count += 1
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / batch_count
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        batch_count = 0
        
        logger.info("\nValidation phase:")
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(valid_loader):
                images = images.to(model.device)
                labels = labels.to(model.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()
                batch_count += 1
                
                if (batch_idx + 1) % 5 == 0:
                    logger.info(f"Validation Batch [{batch_idx + 1}/{len(valid_loader)}]")
        
        avg_valid_loss = valid_loss / batch_count
        valid_accuracy = 100 * valid_correct / valid_total
        
        # Update learning rate
        scheduler.step(avg_valid_loss)
        
        # Store metrics history
        train_history['loss'].append(avg_train_loss)
        train_history['acc'].append(train_accuracy)
        valid_history['loss'].append(avg_valid_loss)
        valid_history['acc'].append(valid_accuracy)
        
        # Print epoch statistics
        logger.info("\nEpoch Summary:")
        logger.info(f"Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")
        logger.info(f"Valid Loss: {avg_valid_loss:.4f} | Valid Accuracy: {valid_accuracy:.2f}%")
        
        # Overfitting detection
        if epoch >= min_epochs:
            acc_gap = train_accuracy - valid_accuracy
            loss_gap = avg_valid_loss - avg_train_loss
            
            if acc_gap > overfitting_threshold * 100:  # Convert threshold to percentage
                consecutive_overfitting += 1
                logger. warning(f"Potential overfitting detected: "
                             f"accuracy gap = {acc_gap:.2f}%, "
                             f"loss gap = {loss_gap:.4f}")
            else:
                consecutive_overfitting = 0
            
            if consecutive_overfitting >= max_overfitting_epochs:
                logger.warning("Stopping due to consistent overfitting")
                break
        
        # Save best model based on validation accuracy
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            best_valid_loss = avg_valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_history': train_history,
                'valid_history': valid_history,
                'best_accuracy': best_accuracy,
                'best_valid_loss': best_valid_loss
            }, 'best_model.pth')
            logger.info(f"Saved new best model with accuracy: {valid_accuracy:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    logger.info("\nTraining completed!")
    logger.info(f"Best validation accuracy: {best_accuracy:.2f}%")
    logger.info(f"Best validation loss: {best_valid_loss:.4f}")
    return model

# Load trained model
def load_model(model_path: str, num_classes: int) -> ProductCNN:
    """
    Load a trained model
    
    Args:
        model_path: Path to the saved model file
        num_classes: Number of product classes
        
    Returns:
        Loaded model
    """
    model = ProductCNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=model.device))
    model.eval()
    return model

# Main function to train the model
def main():
    training_dir = 'training_images'
    num_classes = len([d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d))])
    logger.info(f"Found {num_classes} classes in {training_dir}")

    cnn = ProductCNN(num_classes=num_classes)
    logger.info(f"Training on device: {cnn.device}")

    # Collect images and labels
    image_paths, labels = [], []
    class_mapping = {}

    for idx, class_name in enumerate(sorted(os.listdir(training_dir))):
        class_dir = os.path.join(training_dir, class_name)
        if os.path.isdir(class_dir):
            class_mapping[class_name] = idx
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, img_file))
                    labels.append(idx)

    # Save class mapping
    with open('class_mapping.json', 'w') as f:
        print ("_____________class_mapping_________", )
        json.dump(class_mapping, f)

    # Split dataset
    total_size = len(image_paths)
    train_size = int(0.8 * total_size)
    train_paths, train_labels = image_paths[:train_size], labels[:train_size]
    valid_paths, valid_labels = image_paths[train_size:], labels[train_size:]

    logger.info(f"Total images: {total_size}")
    logger.info(f"Training images: {len(train_paths)}")
    logger.info(f"Validation images: {len(valid_paths)}")

    # Create datasets
    train_dataset = ProductImageDataset(train_paths, train_labels, is_training=True)
    valid_dataset = ProductImageDataset(valid_paths, valid_labels, is_training=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32)

    # Train model
    train_model(cnn, train_loader, valid_loader, num_epochs=50)

    logger.info("Training completed. Model saved as 'best_model.pth'")

if __name__ == "__main__":
    main()