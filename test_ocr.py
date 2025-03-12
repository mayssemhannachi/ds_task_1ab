import base64
from ocr_service import OCRService
import cv2
import numpy as np
from PIL import Image
import io

def test_file_path(ocr_service, image_path):
    """Test OCR with a file path"""
    print("\n1. Testing with file path:")
    print(f"Image path: {image_path}")
    result = ocr_service.process_image(image_path)
    if result['success']:
        print(f"✅ Success! Extracted text: {result['text']}")
    else:
        print(f"❌ Error: {result['error']}")

def test_bytes(ocr_service, image_path):
    """Test OCR with bytes data"""
    print("\n2. Testing with bytes data:")
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        result = ocr_service.process_image(image_bytes)
        if result['success']:
            print(f"✅ Success! Extracted text: {result['text']}")
        else:
            print(f"❌ Error: {result['error']}")
    except FileNotFoundError:
        print(f"❌ Error: File not found: {image_path}")

def test_numpy_array(ocr_service, image_path):
    """Test OCR with numpy array"""
    print("\n3. Testing with numpy array:")
    try:
        # Read image as numpy array using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Failed to load image: {image_path}")
            return
            
        result = ocr_service.process_image(image)
        if result['success']:
            print(f"✅ Success! Extracted text: {result['text']}")
        else:
            print(f"❌ Error: {result['error']}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_base64(ocr_service, image_path):
    """Test OCR with base64 encoded image"""
    print("\n4. Testing with base64 encoded image:")
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        # Decode base64 back to bytes for OCR processing
        image_data = base64.b64decode(base64_image)
        result = ocr_service.process_image(image_data)
        if result['success']:
            print(f"✅ Success! Extracted text: {result['text']}")
        else:
            print(f"❌ Error: {result['error']}")
    except FileNotFoundError:
        print(f"❌ Error: File not found: {image_path}")

def main():
    # Initialize OCR service
    ocr_service = OCRService()
    
    # Test images - you can add more test images here
    test_images = [
    "Screenshot 2023-11-03 at 8.00.52 PM-PhotoRoom.png-PhotoRoom.png", 
    "image.png"  ]
    
    for image_path in test_images:
        print(f"\n{'='*50}")
        print(f"Testing image: {image_path}")
        print('='*50)
        
        # Run all test methods
        test_file_path(ocr_service, image_path)
        test_bytes(ocr_service, image_path)
        test_numpy_array(ocr_service, image_path)
        test_base64(ocr_service, image_path)

if __name__ == "__main__":
    main() 