import pandas as pd
from task1.data_cleaner import DataCleaner

def main():
    # This section reads the dataset from a CSV file and displays information about the dataset.
    print("Reading dataset...")
    df = pd.read_csv('data/dataset/dataset.csv')  # Load the dataset from a CSV file.
    
    print("\nOriginal dataset info:")
    print(df.info())  # Display information about the dataset
    print("\nOriginal dataset sample:")
    print(df.head())  # Display the first 10 rows of the dataset as a sample.
    
    # This section cleans the dataset using the DataCleaner class.
    print("\nCleaning dataset...")
    cleaner = DataCleaner(df)  # Initialize the DataCleaner with the original dataset.
    cleaned_df = cleaner.clean_data()  # Call the clean_data method to clean the dataset.
    
    print("\nCleaned dataset info:")
    print(cleaned_df.info())  # Display information about the cleaned dataset.
    print("\nCleaned dataset sample:")
    print(cleaned_df.head())  # Display the first 10 rows of the cleaned dataset as a sample.
    
    # This section saves the cleaned dataset to a new CSV file.
    cleaned_df.to_csv('cleaned_dataset.csv', index=False)  # Save the cleaned dataset to a CSV file, excluding the index column.
    print("\nCleaned dataset saved to 'cleaned_dataset.csv'")

if __name__ == "__main__":
    main() 