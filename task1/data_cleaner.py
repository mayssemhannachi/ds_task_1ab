import pandas as pd
import numpy as np
import re
from datetime import datetime

class DataCleaner:
    def __init__(self, df):
        """Initialize with a pandas DataFrame"""
        self.df = df.copy()
        
    def clean_text_columns(self):
        """Clean text columns by removing special characters and emojis"""
        # Function to remove emojis, special characters, and normalize text
        def clean_text(text):
            if pd.isna(text):
                return text
            # Remove emojis, special characters, and normalize text
            text = re.sub(r'[^\x00-\x7F]+', '', str(text))
            text = re.sub(r'[^\w\s.,&()-]', '', text)
            # Remove 'xxy' prefix (case insensitive)
            text = re.sub(r'^xxy', '', text, flags=re.IGNORECASE)
            return text.strip()
        
        # Clean Description, Country columns and InvoiceNo
        self.df['Description'] = self.df['Description'].apply(clean_text)
        self.df['Country'] = self.df['Country'].apply(clean_text)
        self.df['StockCode'] = self.df['StockCode'].apply(clean_text)
        self.df['InvoiceNo'] = self.df['InvoiceNo'].apply(clean_text)
        
    def fix_data_types(self):
        """Convert columns to appropriate data types"""
        # Fix Quantity - remove special characters and convert to integer
        self.df['Quantity'] = self.df['Quantity'].apply(lambda x: re.sub(r'[^\d.-]', '', str(x)))
        self.df['Quantity'] = pd.to_numeric(self.df['Quantity'], errors='coerce')
        
        # Fix UnitPrice - remove special characters and convert to float
        self.df['UnitPrice'] = self.df['UnitPrice'].apply(lambda x: re.sub(r'[^\d.-]', '', str(x)))
        self.df['UnitPrice'] = pd.to_numeric(self.df['UnitPrice'], errors='coerce')
        
        # Convert CustomerID to float (allowing for NaN values)
        self.df['CustomerID'] = self.df['CustomerID'].apply(lambda x: re.sub(r'[^\d.]', '', str(x)))
        self.df['CustomerID'] = pd.to_numeric(self.df['CustomerID'], errors='coerce')
        
        # Convert InvoiceDate to datetime
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'], errors='coerce')
        
    def remove_duplicates(self):
        """Remove duplicate entries"""
        self.df.drop_duplicates(inplace=True)
        
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        # Drop rows where Description is missing (as it's crucial for product identification)
        self.df.dropna(subset=['Description'], inplace=True)
        
        # For CustomerID, we'll keep NaN values as they might represent guest purchases
        
        # Remove rows with invalid prices or quantities
        self.df = self.df[self.df['UnitPrice'] > 0]
        self.df = self.df[self.df['Quantity'] != 0]
        
    def standardize_formats(self):
        """Standardize formats across the dataset"""
        # Create a mapping for country names standardization
        country_mapping = {
            'united kingdom': 'United Kingdom',
            'france': 'France'
            # Add more mappings as needed
        }
        
        # Standardize country names
        self.df['Country'] = self.df['Country'].str.lower().map(country_mapping).fillna(self.df['Country'])
        
        # Standardize Description (convert to title case)
        self.df['Description'] = self.df['Description'].str.title()
        
    def clean_data(self):
        """Execute all cleaning steps"""
        self.clean_text_columns()
        self.fix_data_types()
        self.remove_duplicates()
        self.handle_missing_values()
        self.standardize_formats()
        return self.df 