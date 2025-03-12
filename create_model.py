import torch
import torch.nn as nn
from cnn_model import ProductCNN

# Create a simple model with 10 classes for testing
model = ProductCNN(num_classes=10)

# Save the model
torch.save(model.state_dict(), 'models/product_cnn.pth')
print("Created temporary model file for testing") 