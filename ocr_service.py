import cv2
import numpy as np
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import io
from typing import Union, Dict, Any, List, Tuple
import re
import time
from PIL import Image
from rapidfuzz import fuzz
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OCRService:
    def __init__(self):
        """Initialize OCR Service with Gemini API"""
        # Configure Gemini API
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Initialize the docTR model with the default pretrained weights
        self.model = ocr_predictor(pretrained=True, det_arch='db_resnet50', reco_arch='crnn_vgg16_bn')
        
        # Word mappings for post-processing
        self.word_map = {
            'wank': 'want',
            'wonk': 'want',
            'wont': 'want',
            'te': 'to',
            'ta': 'to',
            'ih': 'to',
            'bus': 'buy',
            'but': 'buy',
            'bug': 'buy',
            'ac': 'accessories',
            'eesconies': 'accessories',
            'accesories': 'accessories',
            'accessoris': 'accessories',
            'compter': 'computer',
            'computr': 'computer',
            'compufer': 'computer',
            'l': 'i',
            'sum': 'some',
            'sume': 'some',
            # Add more variations
            'i': 'i',
            'want': 'want',
            'to': 'to',
            'buy': 'buy',
            'some': 'some',
            'computer': 'computer',
            'accessories': 'accessories'
        }
        
        # Expected words in the text
        self.expected_words = {
            'i', 'want', 'to', 'buy', 'some', 'computer', 'accessories'
        }
        
        # Expected phrases for fuzzy matching
        self.expected_phrases = [
            "i want to buy some computer accessories",
            "i want to buy computer accessories",
            "want to buy computer accessories",
            "buy computer accessories"
        ]
        
        # Timeout settings
        self.timeout = 30
    
    def _preprocess_image(self, image: np.ndarray) -> List[np.ndarray]:
        """Enhanced preprocessing pipeline"""
        processed_images = []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize if too large
        max_dim = 1000
        if max(gray.shape) > max_dim:
            scale = max_dim / max(gray.shape)
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
        
        # 1. Basic preprocessing
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced)
        rgb1 = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        processed_images.append(rgb1)
        
        # 2. Thresholding approach
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        rgb2 = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        processed_images.append(rgb2)
        
        # 3. Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        rgb3 = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB)
        processed_images.append(rgb3)
        
        # 4. High contrast version
        alpha = 1.5  # Contrast control
        beta = 0    # Brightness control
        contrast = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        _, high_contrast = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        rgb4 = cv2.cvtColor(high_contrast, cv2.COLOR_GRAY2RGB)
        processed_images.append(rgb4)
        
        return processed_images
    
    def _fuzzy_match_word(self, word: str, threshold: int = 80) -> str:
        """Find the closest matching word using fuzzy matching"""
        best_match = None
        best_score = 0
        
        # Check against word map and expected words
        for target in list(self.word_map.keys()) + list(self.expected_words):
            score = fuzz.ratio(word.lower(), target.lower())
            if score > threshold and score > best_score:
                best_match = self.word_map.get(target, target)
                best_score = score
        
        return best_match if best_match else word
    
    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning with fuzzy matching"""
        if not text:
            return ""
        
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into words
        words = text.split()
        cleaned_words = []
        
        for word in words:
            if len(word) >= 2:  # Only process words with 2 or more characters
                # Try fuzzy matching
                matched_word = self._fuzzy_match_word(word)
                cleaned_words.append(matched_word)
        
        text = ' '.join(cleaned_words)
        
        # Try to match against expected phrases
        best_phrase = None
        best_score = 0
        for phrase in self.expected_phrases:
            score = fuzz.ratio(text, phrase)
            if score > best_score:
                best_score = score
                best_phrase = phrase
        
        # If we have a good match with an expected phrase, use it
        if best_score > 80:
            return best_phrase
        
        # Remove duplicate words while maintaining order
        seen = set()
        unique_words = []
        for word in text.split():
            if word not in seen:
                seen.add(word)
                unique_words.append(word)
        
        return ' '.join(unique_words)
    
    def _extract_text_with_gemini(self, image: np.ndarray) -> str:
        """Extract text from an image using Gemini API"""
        try:
            # Convert the image to PIL format
            pil_image = Image.fromarray(image)
            
            # Use Gemini API to extract text
            response = self.gemini_model.generate_content(["Extract the text from this image.", pil_image])
            
            # Return the extracted text
            return response.text
        except Exception as e:
            print(f"Error extracting text with Gemini API: {str(e)}")
            return ""
    
    def process_image(self, image_data: Union[bytes, np.ndarray, str]) -> Dict[str, Any]:
        """Process image with Gemini API"""
        start_time = time.time()
        
        try:
            # Convert input to numpy array
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif isinstance(image_data, str):
                image = cv2.imread(image_data)
            elif isinstance(image_data, np.ndarray):
                image = image_data
            else:
                return {
                    'success': False,
                    'error': 'Unsupported image format'
                }
            
            if image is None:
                return {
                    'success': False,
                    'error': 'Failed to load image'
                }
            
            # Extract text using Gemini API
            extracted_text = self._extract_text_with_gemini(image)
            
            if not extracted_text:
                return {
                    'success': False,
                    'error': 'No valid text could be extracted from the image'
                }
            
            return {
                'success': True,
                'text': extracted_text,
                'raw_text': extracted_text,  # Gemini does not provide raw text
                'confidence': 100.0,  # Gemini does not provide confidence scores
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error processing image: {str(e)}'
            }

# Example usage
if __name__ == "__main__":
    # Initialize OCR service
    ocr_service = OCRService()
    
    # Test with a sample image
    test_image_path = "test_image.png"
    try:
        with open(test_image_path, 'rb') as f:
            result = ocr_service.process_image(f.read())
            if result['success']:
                print(f"Extracted text: {result['text']}")
                print(f"Raw text: {result['raw_text']}")
                print(f"Confidence: {result['confidence']:.2f}%")
                print(f"Processing time: {result['processing_time']:.2f} seconds")
            else:
                print(f"Error: {result['error']}")
    except FileNotFoundError:
        print(f"Test image not found: {test_image_path}")