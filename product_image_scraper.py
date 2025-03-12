import os
import csv
import requests
from bs4 import BeautifulSoup
import logging
import time
from typing import List, Dict
from urllib.parse import quote_plus, urlencode
from PIL import Image
from io import BytesIO
import random
import pandas as pd
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
import colorama
from colorama import Fore, Style
import numpy as np
from PIL import ImageEnhance, ImageOps

class ProductImageScraper:
    def __init__(self, 
                 stock_codes_path: str = 'data/dataset/CNN_Model_Train_Data.csv',
                 dataset_path: str = 'data/dataset/cleaned_dataset.csv',
                 output_dir: str = 'training_images',
                 max_images_per_product: int = 50):  # Increased to 50 images
        
        # Initialize colorama for colored console output
        colorama.init()
        
        self.stock_codes_path = stock_codes_path
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.max_images_per_product = max_images_per_product
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scraping.log'),
                logging.StreamHandler()
            ]
        )
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize session with rotating user agents
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        
        # Search URLs for multiple sources
        self.base_urls = {
            'amazon': 'https://www.amazon.com/s?k={}&i=kitchen',
            'walmart': 'https://www.walmart.com/search?q={}',
            'target': 'https://www.target.com/s?searchTerm={}'
        }
        
        # Load product data
        self._load_datasets()
        
    def _setup_webdriver(self):
        """Setup Selenium WebDriver with Chrome"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920x1080')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            
            # Add random user agent
            chrome_options.add_argument(f'user-agent={random.choice(self.user_agents)}')
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.set_page_load_timeout(20)
            return driver
            
        except Exception as e:
            logging.error(f"Error setting up WebDriver: {str(e)}")
            raise
            
    def _load_datasets(self):
        """Load and process the datasets"""
        try:
            # Load stock codes
            stock_codes_df = pd.read_csv(self.stock_codes_path)
            self.stock_codes = []
            
            # Clean stock codes
            for code in stock_codes_df['StockCode']:
                cleaned_code = str(code).strip().replace('ö', '').replace('^', '')
                if cleaned_code:
                    self.stock_codes.append(cleaned_code)
            
            # Load product descriptions
            self.descriptions_df = pd.read_csv(self.dataset_path)
            
            logging.info(f"Loaded {len(self.stock_codes)} products from {self.stock_codes_path}")
            
        except Exception as e:
            logging.error(f"Error loading datasets: {str(e)}")
            raise
            
    def _get_product_description(self, stock_code: str) -> str:
        """Get product description for a stock code"""
        try:
            # Find the product in the cleaned dataset
            product = self.descriptions_df[
                self.descriptions_df['StockCode'].astype(str).str.strip() == stock_code
            ].iloc[0]
            
            return product['Description']
        except:
            return f"Product {stock_code}"
            
    def _augment_image(self, img: Image.Image, index: int, output_dir: str, stock_code: str) -> None:
        """Apply data augmentation to create additional training images"""
        try:
            # Base transformations
            transforms = [
                ('original', lambda x: x),
                ('rotate_90', lambda x: x.rotate(90)),
                ('rotate_180', lambda x: x.rotate(180)),
                ('rotate_270', lambda x: x.rotate(270)),
                ('flip_lr', lambda x: ImageOps.mirror(x)),
                ('flip_ud', lambda x: ImageOps.flip(x)),
                ('brightness_up', lambda x: ImageEnhance.Brightness(x).enhance(1.3)),
                ('brightness_down', lambda x: ImageEnhance.Brightness(x).enhance(0.7)),
                ('contrast_up', lambda x: ImageEnhance.Contrast(x).enhance(1.3)),
                ('contrast_down', lambda x: ImageEnhance.Contrast(x).enhance(0.7))
            ]
            
            # Apply each transformation and save
            for transform_name, transform_func in transforms:
                try:
                    transformed_img = transform_func(img)
                    output_path = os.path.join(output_dir, f"{index}_{transform_name}.jpg")
                    transformed_img.convert('RGB').save(output_path, 'JPEG', quality=95)
                except Exception as e:
                    print(f"{Fore.RED}Failed to apply {transform_name} transformation: {str(e)}{Style.RESET_ALL}")
                    
        except Exception as e:
            print(f"{Fore.RED}Error in image augmentation: {str(e)}{Style.RESET_ALL}")

    def _download_image(self, url: str, stock_code: str, index: int) -> bool:
        """Download and save an image with augmentation"""
        try:
            headers = {'User-Agent': random.choice(self.user_agents)}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # Verify it's an image
                img = Image.open(BytesIO(response.content))
                
                # Check image dimensions and quality
                width, height = img.size
                if width < 300 or height < 300:  # Minimum size check
                    return False
                
                # Create output directory
                output_dir = os.path.join(self.output_dir, stock_code)
                os.makedirs(output_dir, exist_ok=True)
                
                # Save original image
                output_path = os.path.join(output_dir, f"{index}.jpg")
                img.convert('RGB').save(output_path, 'JPEG', quality=95)
                
                # Apply augmentation
                self._augment_image(img, index, output_dir, stock_code)
                
                print(f"{Fore.GREEN}✓ Downloaded and augmented image {index}/{self.max_images_per_product} for product {stock_code}{Style.RESET_ALL}")
                return True
                
        except Exception as e:
            print(f"{Fore.RED}✗ Failed to download image: {str(e)}{Style.RESET_ALL}")
            
        return False
        
    def _get_search_terms(self, stock_code: str) -> List[str]:
        """Generate more specific search terms for better matches"""
        description = self._get_product_description(stock_code)
        
        # Clean description and extract key terms
        description = re.sub(r'[^\w\s]', ' ', description)
        words = description.split()
        
        # Product categories and specific terms
        categories = {
            'clock': ['alarm clock', 'desk clock', 'vintage clock', 'retro clock'],
            'bag': ['lunch bag', 'tote bag', 'shopping bag', 'carry bag'],
            'bottle': ['water bottle', 'drink bottle', 'beverage container'],
            'box': ['storage box', 'container', 'organizer'],
            'stand': ['display stand', 'holder', 'rack']
        }
        
        # Find matching categories
        matching_terms = []
        desc_lower = description.lower()
        for category, specific_terms in categories.items():
            if category in desc_lower:
                matching_terms.extend(specific_terms)
        
        # Generate base search terms
        search_terms = [
            description,
            ' '.join(words[:3]),
            f"vintage {description}",
            f"retro {description}",
            f"{description} home decor"
        ]
        
        # Add category-specific terms
        if matching_terms:
            for term in matching_terms:
                search_terms.append(f"{term} {' '.join(words[:2])}")
        
        # Add color and style variations
        colors = ['red', 'blue', 'green', 'white', 'black', 'pink', 'yellow', 'brown']
        styles = ['vintage', 'retro', 'modern', 'classic', 'traditional']
        
        for word in words:
            if word.lower() in colors:
                for style in styles:
                    search_terms.append(f"{style} {word} {description}")
        
        # Remove duplicates while preserving order
        search_terms = list(dict.fromkeys(search_terms))
        
        # Print search terms in color
        print(f"\n{Fore.CYAN}Search terms for stock code {stock_code}:{Style.RESET_ALL}")
        for i, term in enumerate(search_terms, 1):
            print(f"{Fore.GREEN}{i}. {term}{Style.RESET_ALL}")
        
        return search_terms
        
    def _extract_images_from_page(self, driver) -> List[str]:
        """Extract image URLs from Amazon search results"""
        image_urls = []
        try:
            # Wait for product images to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "img.s-image"))
            )
            
            # Scroll multiple times to load more images
            for _ in range(3):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1)
            
            # Get all product images
            images = driver.find_elements(By.CSS_SELECTOR, "img.s-image")
            
            # Get high-resolution image URLs
            for img in images:
                try:
                    src = img.get_attribute('src')
                    if src and src.startswith('http'):
                        # Try to get higher resolution image
                        high_res_src = src.replace('_AC_US40_', '_AC_SL1500_').replace('_AC_UL320_', '_AC_SL1500_')
                        image_urls.append(high_res_src)
                except:
                    continue
            
            print(f"{Fore.YELLOW}Found {len(image_urls)} product images{Style.RESET_ALL}")
            
        except TimeoutException:
            print(f"{Fore.RED}Timeout waiting for images to load{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error extracting images: {str(e)}{Style.RESET_ALL}")
            
        return image_urls
        
    def _scrape_product_images(self, driver, search_term: str) -> List[str]:
        """Scrape product images from Amazon"""
        urls = []
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                print(f"\n{Fore.BLUE}Searching Amazon for: {search_term} (Attempt {attempt + 1}/{max_retries}){Style.RESET_ALL}")
                
                # Construct search URL with department filter
                search_url = self.base_urls['amazon'].format(quote_plus(search_term))
                
                # Load page with Selenium
                driver.get(search_url)
                
                # Extract image URLs
                urls = self._extract_images_from_page(driver)
                
                if urls:  # If we found images, break the retry loop
                    break
                    
                # Add longer delay between retries
                time.sleep(random.uniform(5, 8))
                
            except WebDriverException:
                print(f"{Fore.RED}WebDriver error, retrying...{Style.RESET_ALL}")
                time.sleep(random.uniform(10, 15))  # Longer delay after error
                continue
                
            except Exception as e:
                print(f"{Fore.RED}Error searching Amazon: {str(e)}{Style.RESET_ALL}")
                break
                
        return urls
        
    def _process_product(self, stock_code: str) -> int:
        """Process a single product"""
        images_downloaded = 0
        search_terms = self._get_search_terms(stock_code)
        
        print(f"\n{Fore.CYAN}Processing product {stock_code}{Style.RESET_ALL}")
        
        # Create a new WebDriver instance for each product
        driver = self._setup_webdriver()
        
        try:
            for search_term in search_terms:
                if images_downloaded >= self.max_images_per_product:
                    break
                    
                urls = self._scrape_product_images(driver, search_term)
                
                # Try to download each image
                for url in urls:
                    if images_downloaded >= self.max_images_per_product:
                        break
                        
                    if self._download_image(url, stock_code, images_downloaded + 1):
                        images_downloaded += 1
                        time.sleep(random.uniform(1, 2))  # Small delay between image downloads
                        
                # Add delay between searches
                time.sleep(random.uniform(3, 5))
                
            if images_downloaded == 0:
                print(f"{Fore.RED}No images found for stock code: {stock_code}{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}Successfully downloaded {images_downloaded} images for {stock_code}{Style.RESET_ALL}")
                
        finally:
            # Always close the driver
            try:
                driver.quit()
            except:
                pass
            
        return images_downloaded
        
    def process_products(self):
        """Process all products sequentially"""
        total_products = len(self.stock_codes)
        successful_products = 0
        
        print(f"\n{Fore.CYAN}Starting sequential image scraping for {total_products} products from Amazon{Style.RESET_ALL}")
        
        for i, stock_code in enumerate(self.stock_codes, 1):
            print(f"\n{Fore.CYAN}Processing product {i}/{total_products}: {stock_code}{Style.RESET_ALL}")
            
            try:
                images_downloaded = self._process_product(stock_code)
                if images_downloaded > 0:
                    successful_products += 1
                    
                # Add longer delay between products
                if i < total_products:  # Don't delay after the last product
                    delay = random.uniform(8, 12)
                    print(f"{Fore.YELLOW}Waiting {delay:.1f} seconds before next product...{Style.RESET_ALL}")
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"{Fore.RED}Error processing stock code {stock_code}: {str(e)}{Style.RESET_ALL}")
                continue
                
        print(f"\n{Fore.GREEN}Successfully downloaded images for {successful_products}/{total_products} products{Style.RESET_ALL}")

def main():
    """Main function to run the scraper"""
    scraper = ProductImageScraper(
        stock_codes_path='data/dataset/CNN_Model_Train_Data.csv',
        dataset_path='data/dataset/cleaned_dataset.csv',
        output_dir='training_images',
        max_images_per_product=50  # Increased to 50 images
    )
    
    print(f"{Fore.CYAN}Starting enhanced image scraping process...{Style.RESET_ALL}")
    scraper.process_products()
    print(f"{Fore.GREEN}Image scraping and augmentation completed!{Style.RESET_ALL}")

if __name__ == "__main__":
    main() 