from flask import Flask, request, jsonify, render_template, flash, redirect, url_for, send_from_directory
from product_recommendation_service import ProductRecommendationService
from ocr_service import OCRService
import base64
import io
from PIL import Image
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from cnn_model import ProductCNN, load_model
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import json
import torch



# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for flash messages

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize services
recommendation_service = ProductRecommendationService(
    api_key=os.getenv('PINECONE_API_KEY')
)
ocr_service = OCRService()

# Load class mapping from CSV
def load_class_mapping():
    try:
        # Read both CSV files
        logger.info("Reading model dataset CSV...")
        model_df = pd.read_csv('data/dataset/CNN_Model_Train_Data.csv')
        logger.info(f"Model dataset loaded with {len(model_df)} rows")
        
        logger.info("Reading details dataset CSV...")
        details_df = pd.read_csv('data/dataset/cleaned_dataset.csv')
        logger.info(f"Details dataset loaded with {len(details_df)} rows")
        
        if model_df.empty:
            logger.error("Empty model dataset loaded")
            return {}
            
        # Create a mapping of unique stock codes with their descriptions
        stock_codes = []
        stock_code_details = {}
        
        # First, get all valid stock codes from the model training data
        logger.info("Processing stock codes from model dataset...")
        for _, row in model_df.iterrows():
            try:
                # Clean the stock code by removing special characters
                code = str(row['StockCode']).strip()
                code = code.replace('ö', '').replace('^', '')
                if code and code not in stock_codes:
                    stock_codes.append(code)
                    logger.info(f"Added stock code: {code}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping row due to error: {e}")
                continue
        
        logger.info(f"Found {len(stock_codes)} unique stock codes")
        
        # Then, get details for each stock code from the cleaned dataset
        logger.info("Loading product details...")
        for code in stock_codes:
            try:
                # Find matching row in details dataset
                # Convert both to string and clean for comparison
                details_df['StockCode'] = details_df['StockCode'].astype(str).str.strip()
                product = details_df[details_df['StockCode'] == code].iloc[0]
                stock_code_details[code] = {
                    'description': str(product['Description']).strip(),
                    'country': str(product['Country']).strip(),
                    'unit_price': float(product['UnitPrice'])
                }
                logger.info(f"Found details for stock code {code}")
            except Exception as e:
                logger.warning(f"Could not find details for stock code {code}: {e}")
                stock_code_details[code] = {
                    'description': f'Product {code}',
                    'country': 'Unknown',
                    'unit_price': 0.00
                }
        
        if not stock_codes:
            logger.error("No valid stock codes found in dataset")
            return {}
            
        # Create the class mapping
        class_mapping = {idx: code for idx, code in enumerate(stock_codes)}
        logger.info(f"Created class mapping with {len(class_mapping)} entries")
        logger.info(f"Class mapping: {class_mapping}")
        
        # Store the details mapping globally
        global PRODUCT_DETAILS
        PRODUCT_DETAILS = stock_code_details
        logger.info(f"Stored {len(PRODUCT_DETAILS)} product details")
        
        return class_mapping
        
    except Exception as e:
        logger.error(f"Error loading class mapping: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {}

# Initialize global product details
PRODUCT_DETAILS = {}

# Load class mapping
CLASS_MAPPING = load_class_mapping()
NUM_CLASSES = len(CLASS_MAPPING)

logger.info(f"Loaded class mapping with {NUM_CLASSES} classes")
logger.info(f"Class mapping: {CLASS_MAPPING}")

if not CLASS_MAPPING:
    logger.error("Failed to load class mapping, product classifier will not be initialized")
    product_classifier = None
else:
    try:
        # Initialize the CNN model
        logger.info("Initializing CNN model...")
        product_classifier = ProductCNN(num_classes=NUM_CLASSES)
        logger.info(f"CNN model created on device: {product_classifier.device}")
        
        # Load the saved model state
        logger.info("Loading model state from best_model.pth...")
        checkpoint = torch.load('best_model.pth', map_location=product_classifier.device)
        logger.info("Model state loaded successfully")
        
        logger.info("Loading state dictionary into model...")
        product_classifier.load_state_dict(checkpoint)  # Load directly from checkpoint
        logger.info("State dictionary loaded successfully")
        
        product_classifier.eval()  # Set to evaluation mode
        logger.info("Model set to evaluation mode")
        logger.info(f"Product classifier initialized successfully with {NUM_CLASSES} classes")
        
        # Save the class mapping to a file for the model to use
        with open('class_mapping.json', 'w') as f:
            json.dump(CLASS_MAPPING, f)
        logger.info("Saved class mapping to class_mapping.json")
        
    except Exception as e:
        logger.error(f"Error initializing product classifier: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        product_classifier = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Landing page - redirects to text query interface"""
    return redirect(url_for('text_query'))

@app.route('/text-query', methods=['GET', 'POST'])
def text_query():
    """Handle text-based product queries"""
    if request.method == 'POST':
        query = request.form.get('query')
        if not query:
            flash('Please enter a query', 'warning')
            return redirect(url_for('text_query'))
        
        try:
            results = recommendation_service.get_recommendations(query)
            return render_template('text_query.html', response=results)
        except Exception as e:
            flash(f'Error processing query: {str(e)}', 'danger')
            return redirect(url_for('text_query'))
    
    return render_template('text_query.html')

@app.route('/image-query', methods=['GET', 'POST'])
def image_query():
    """
    Endpoint for image-based product detection
    
    GET: Render the image upload page
    POST: Process the uploaded image and return results
    """
    if request.method == 'POST':
        try:
            # Check if image was uploaded
            if 'image' not in request.files:
                return render_template('image_query.html', error='No image file provided')
                
            file = request.files['image']
            if file.filename == '':
                return render_template('image_query.html', error='No selected file')
                
            if not allowed_file(file.filename):
                return render_template('image_query.html', error='Invalid file type')
                
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image with Gemini API
            ocr_result = ocr_service.process_image(filepath)
            
            if not ocr_result['success']:
                raise ValueError(f"OCR processing failed: {ocr_result.get('error', 'Unknown error')}")
            
            # Get the extracted text
            extracted_text = ocr_result['text']
            
            # Get product recommendations based on extracted text
            recommendations = recommendation_service.get_recommendations(extracted_text)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            # Render the results in a template
            return render_template('image_query.html',
                                   extracted_text=extracted_text,
                                   response={
                                       'text': extracted_text,
                                       'raw_text': ocr_result.get('raw_text', ''),
                                       'confidence': ocr_result.get('confidence', 0.0),
                                       'recommendations': recommendations
                                   })
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return render_template('image_query.html', error=str(e))
    
    # GET request - render the image upload page
    return render_template('image_query.html')

def get_product_details(stock_code):
    """Get product details from the global mapping"""
    try:
        if stock_code in PRODUCT_DETAILS:
            return PRODUCT_DETAILS[stock_code]
        else:
            logger.warning(f"No product found with stock code: {stock_code}")
            return {
                'description': 'Product details not found',
                'country': 'Unknown',
                'unit_price': 0.00
            }
    except Exception as e:
        logger.warning(f"Could not fetch details for product {stock_code}: {str(e)}")
        return {
            'description': 'Error fetching product details',
            'country': 'Unknown',
            'unit_price': 0.00
        }

@app.route('/product-upload', methods=['GET', 'POST'])
def product_upload():
    """Handle product image uploads for identification"""
    if request.method == 'POST':
        if product_classifier is None:
            flash('Product classifier is not initialized. Please check the logs for errors.', 'danger')
            return redirect(url_for('product_upload'))
            
        if 'image' not in request.files:
            flash('No image file uploaded', 'warning')
            return redirect(url_for('product_upload'))
        
        image_file = request.files['image']
        if image_file.filename == '':
            flash('No image selected', 'warning')
            return redirect(url_for('product_upload'))
            
        if not allowed_file(image_file.filename):
            flash('Invalid file type. Allowed types: ' + ', '.join(ALLOWED_EXTENSIONS), 'warning')
            return redirect(url_for('product_upload'))
        
        try:
            # Save the uploaded file
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print("-----filepath-----", filepath)
            image_file.save(filepath)
            logger.info(f"File saved to {filepath}")
            
            # Get predictions using CNN model
            predicted_idx, confidence = product_classifier.predict(filepath)
            logger.info(f"Predicted index: {predicted_idx}, Type: {type(predicted_idx)}")
            
            if predicted_idx is None:
                raise ValueError("Model prediction failed")
            
            # Extract the number from "Unknown class 7"
            if isinstance(predicted_idx, str) and predicted_idx.startswith("Unknown class"):
                try:
                    predicted_idx = int(predicted_idx.split()[-1])  # Extract the last part and convert to integer
                except (ValueError, IndexError):
                    logger.error(f"Failed to extract index from: {predicted_idx}")
                    raise ValueError(f"Invalid predicted index format: {predicted_idx}")
            
            # Ensure predicted_idx is an integer
            try:
                predicted_idx = int(predicted_idx)  # Convert to integer if possible
            except (ValueError, TypeError):
                logger.error(f"Invalid predicted index: {predicted_idx}")
                raise ValueError(f"Invalid predicted index: {predicted_idx}")
            
            # Check if predicted index is in CLASS_MAPPING
            if predicted_idx not in CLASS_MAPPING:
                logger.error(f"Predicted index {predicted_idx} is not in CLASS_MAPPING")
                raise ValueError(f"Invalid predicted index: {predicted_idx}")
            
            # Get the stock code from class mapping
            stock_code = CLASS_MAPPING[predicted_idx]
            logger.info(f"Stock code: {stock_code}")
            
            # Get product details
            details = PRODUCT_DETAILS.get(stock_code, {
                'description': f'Product {stock_code}',
                'country': 'Unknown',
                'unit_price': 0.00
            })
            
            # Get similar products using vector database
            similar_products = recommendation_service.get_recommendations(details['description'])
            
            # Prepare the response
            product_info = {
                'stock_code': stock_code,
                'confidence': confidence,
                'description': details['description'],
                'country': details['country'],
                'unit_price': details['unit_price'],
                'cnn_class': predicted_idx  # Add the CNN class index
            }
            
            # Clean up the uploaded file
            #os.remove(filepath)
            print("filename", filename)

            print("filepath", filepath)
            
            return render_template('product_upload.html',
                                product_info=product_info,
                                similar_products=similar_products,
                                filepath=filepath)
            
        except Exception as e:
            logger.error(f"Error processing product image: {str(e)}")
            # Clean up the uploaded file if it exists
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            flash(f'Error processing product image: {str(e)}', 'danger')
            return redirect(url_for('product_upload'))
    
    return render_template('product_upload.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@app.route('/api/detect_product', methods=['POST'])
def detect_product():
    """
    Endpoint for detecting products from uploaded images.
    Expects a file upload with key 'image'.
    Returns product details and recommendations.
    """
    if product_classifier is None:
        return jsonify({
            'error': 'Product classifier not initialized'
        }), 500

    # Check if image was uploaded
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image file provided'
        }), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({
            'error': 'No selected file'
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type. Allowed types: ' + ', '.join(ALLOWED_EXTENSIONS)
        }), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get predictions
        predicted_idx, confidence = product_classifier.predict(filepath)
        
        if predicted_idx is None:
            raise ValueError("Model prediction failed")
        
        # Get the stock code from class mapping
        stock_code = CLASS_MAPPING[predicted_idx]
        
        # Clean up the uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'product': {
                'class_name': stock_code,
                'confidence': float(confidence)
            },
            'message': f'Product detected: {stock_code} with {confidence:.2%} confidence'
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({
            'error': f'Error processing image: {str(e)}'
        }), 500



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
