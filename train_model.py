import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from cnn_model import ProductCNN, ProductImageDataset, train_model
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_stock_code(code):
    """Clean stock code by removing special characters"""
    return str(code).replace('ö', '').replace('^', '')

def prepare_dataset(data_dir: str, csv_file: str):
    """
    Prepare dataset for training
    
    Args:
        data_dir: Directory containing the training images
        csv_file: Path to the CSV file containing product information
    """
    # Read CSV file
    logger.info(f"Reading CSV file from {csv_file}")
    df = pd.read_csv(csv_file)
    logger.info(f"Found {len(df)} products in CSV file")
    
    # Clean stock codes in the dataframe
    df['StockCode'] = df['StockCode'].apply(clean_stock_code)
    
    # Get all image paths and labels
    image_paths = []
    labels = []
    class_mapping = {}
    
    # Create class mapping
    unique_stock_codes = df['StockCode'].unique()
    logger.info(f"Found {len(unique_stock_codes)} unique stock codes")
    for idx, code in enumerate(unique_stock_codes):
        class_mapping[code] = idx
    
    # Save class mapping
    with open('class_mapping.json', 'w') as f:
        json.dump(class_mapping, f)
    
    # Get all images and their labels
    for code in unique_stock_codes:
        code_dir = os.path.join(data_dir, str(code))
        logger.info(f"Processing directory {code_dir}")
        if os.path.exists(code_dir):
            img_count = 0
            for img_file in os.listdir(code_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(code_dir, img_file))
                    labels.append(class_mapping[code])
                    img_count += 1
            logger.info(f"Found {img_count} images for stock code {code}")
        else:
            logger.warning(f"Directory not found for stock code {code}")
    
    logger.info(f"Total images found: {len(image_paths)}")
    return image_paths, labels, len(unique_stock_codes)

def main():
    # Parameters
    data_dir = 'training_images'
    csv_file = 'data/dataset/CNN_Model_Train_Data_images_scraped.csv'
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    
    # Prepare dataset
    image_paths, labels, num_classes = prepare_dataset(data_dir, csv_file)
    
    if len(image_paths) == 0:
        logger.error("No images found in the dataset. Please check the data directory and CSV file.")
        return
    
    # Split dataset
    total_size = len(image_paths)
    train_size = int(0.8 * total_size)
    valid_size = total_size - train_size
    
    train_paths = image_paths[:train_size]
    train_labels = labels[:train_size]
    valid_paths = image_paths[train_size:]
    valid_labels = labels[train_size:]
    
    logger.info(f"Training set size: {len(train_paths)}")
    logger.info(f"Validation set size: {len(valid_paths)}")
    
    # Initialize model
    model = ProductCNN(num_classes=num_classes)
    
    # Create data loaders
    train_dataset = ProductImageDataset(train_paths, train_labels, transform=model.transform)
    valid_dataset = ProductImageDataset(valid_paths, valid_labels, transform=model.transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    
    # Train model
    train_model(model, train_loader, valid_loader, num_epochs=num_epochs, learning_rate=learning_rate)
    
    logger.info("Training completed. Model saved as 'best_model.pth'")

if __name__ == "__main__":
    main() 