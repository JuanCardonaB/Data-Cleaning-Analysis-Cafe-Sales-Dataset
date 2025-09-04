import logging
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from pathlib import Path
import warnings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)

# Structure of the dataset
# "Transaction ID" | "Item" | "Quantity" | "Price Per Unit" | "Total Spent" | "Payment Method" | "Location" | "Transaction Date"
class CafeSalesDataCleaner:
    # Class for cleaning cafe sales data
    def __init__(self, filepath: str):
        self.file_path = Path(filepath)
        self.df = None
        self.cleaning_report = {}

        self.valid_payment_methods = {'Cash', 'Credit Card', 'Digital Wallet', 'Unknown'}


    def load_data_csv(self) -> pd.DataFrame:
        # Load data from CSV file
        try:
            self.df = pd.read_csv(self.file_path)
            logger.info(f"Data loaded successfully from {self.file_path}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def generate_initial_report(self) -> Dict[str, Any]:
        report = {
            "num_rows": len(self.df),
            "num_columns": len(self.df.columns),
            "missing_values": self.df.isnull().sum().to_dict(),
            "data_types": self.df.dtypes.to_dict(),
            "duplicates": self.df.duplicated().sum()
        }

        logger.info("Initial data report generated")
        return report
    
    def clean_transaction_id(self):
        # Cleans the 'Transaction ID' column
        # Rule: Must be unique and non-null
        initial_null_count = self.df['Transaction ID'].isnull().sum()

        if initial_null_count > 0:
            logger.info(f"Found {initial_null_count} null Transaction IDs.")
            self.df = self.df.dropna(subset=['Transaction ID']) # Remove rows with null Transaction ID
        
        # Verify duplicates
        duplicates = self.df['Transaction ID'].duplicated().sum()
        if duplicates > 0:
            logger.info(f"Found {duplicates} duplicate Transaction IDs. Removing duplicates.")
            self.df = self.df.drop_duplicates(subset=['Transaction ID'])

        self.cleaning_report["Transaction ID"] = {
            "nulls_removed": initial_null_count,
            "duplicates_removed": duplicates
        }

        logger.info("Transaction ID column cleaned.")

    def clean_item(self):
        # Cleans the 'Item' column
        # Rules: 
        # - Missing values → "Unknown Item"
        # - "UNKNOWN" → "Unknown Item"
        # - Standardize values (consistent capitalization)

        initial_null_count = self.df['Item'].isnull().sum()
        logger.info(f"Found {initial_null_count} null Item entries.")

        # Replace missing values
        self.df['Item'] = self.df['Item'].fillna('Unknown Item')

        # Replace "UNKNOWN" with "Unknown Item"
        unknown_count = (self.df['Item'].str.upper() == 'UNKNOWN').sum()
        self.df['Item'] = self.df['Item'].str.upper().replace('UNKNOWN', 'Unknown Item')

        # Standardize capitalization
        self.df['Item'] = self.df['Item'].apply(
            lambda x: x.title() if isinstance(x, str) and x != 'Unknown Item' else x
        )

        self.cleaning_report["Item"] = {
            'nulls_filled': initial_null_count,
            'unknown_replaced': unknown_count,
            'unique_items': self.df['Item'].nunique()
        }

        logger.info("Item column cleaned.")

    def clean_quantity(self):
        # Cleans the 'Quantity' column
        # Rules:
        # - Missing values → impute with median
        # - Convert to int
        initial_null_count = self.df['Quantity'].isnull().sum()

        # # Convert to numeric, coercing errors to NaN
        self.df['Quantity'] = pd.to_numeric(self.df['Quantity'], errors='coerce')

        # Calculate median
        median_quantity = self.df['Quantity'].median()

        # Fill missing values with median
        self.df['Quantity'] = self.df['Quantity'].fillna(median_quantity)

        # Convert to integer
        self.df['Quantity'] = self.df['Quantity'].astype(int)

        self.cleaning_report["Quantity"] = {
            'nulls_filled': initial_null_count,
            'median_used': median_quantity
        }

        logger.info("Quantity column cleaned.")

    def clean_price_per_unit(self):
        # Cleans the 'Price Per Unit' column
        # Rules:
        # - Missing values → impute with median per product
        # - Convert to float
        initial_null_count = self.df['Price Per Unit'].isnull().sum()

        # Convert to numeric, coercing errors to NaN
        self.df['Price Per Unit'] = pd.to_numeric(self.df['Price Per Unit'], errors='coerce')

        # Calculate median per item
        median_by_item = self.df.groupby('Item')['Price Per Unit'].median()

        def impute_price(row):
            if pd.isnull(row['Price Per Unit']):
                return median_by_item.get(row['Item'], self.df['Price Per Unit'].median())
            return row['Price Per Unit']
        
        self.df['Price Per Unit'] = self.df.apply(impute_price, axis=1)

        # Convert to float
        self.df['Price Per Unit'] = self.df['Price Per Unit'].astype(float)

        self.cleaning_report["Price Per Unit"] = {
            'nulls_filled': initial_null_count,
             'median_by_item_used': True
        }

        logger.info("Price Per Unit column cleaned.")

    def clean_total_spent(self):
        # Cleans the 'Total Spent' column
        # Rules:
        # - Missing values and "ERROR" → recalculate as Quantity * Price Per Unit
        initial_null_count = self.df['Total Spent'].isnull().sum()
        error_count = (self.df['Total Spent'] == 'ERROR').sum() if 'Total Spent' in self.df.columns else 0

        # Recalculate Total Spent
        self.df['Total Spent'] = self.df['Quantity'] * self.df['Price Per Unit']

        self.cleaning_report["Total Spent"] = {
            'initial_nulls': initial_null_count,
            'errors_replaced': error_count,
            'recalculated_all': True
        }

        logger.info("Total Spent column cleaned.")

    def clean_payment_method(self):
        # Cleans the 'Payment Method' column
        # Rules:
        # - Missing values and "ERROR" → "Unknown"
        initial_null_count = self.df['Payment Method'].isnull().sum()
        error_count = (
            (self.df['Payment Method'].str.upper() == 'ERROR') |
            (self.df['Payment Method'].str.upper() == 'UNKNOWN')
        ).sum()

        # Replace missing values
        self.df['Payment Method'] = self.df['Payment Method'].fillna('Unknown')

        # Replace problematic values
        self.df['Payment Method'] = self.df['Payment Method'].str.upper().replace({
            'ERROR': 'Unknown',
            'UNKNOWN': 'Unknown'
        })

        # Standardize capitalization
        self.df['Payment Method'] = self.df['Payment Method'].apply(
            lambda x: x.title() if isinstance(x, str) and x != 'Unknown' else x
        )

        # Verify only valid methods remain
        invalid_methods = set(self.df['Payment Method'].unique()) - self.valid_payment_methods
        if invalid_methods:
            logger.warning(f"Found invalid payment methods: {invalid_methods}. Replacing with 'Unknown'.")
            self.df.loc[self.df['Payment Method'].isin(invalid_methods), 'Payment Method'] = 'Unknown'

        self.cleaning_report["Payment Method"] = {
            'nulls_filled': initial_null_count,
            'errors_replaced': error_count,
            'invalid_methods_replaced': len(invalid_methods)
        }

        logger.info("Payment Method column cleaned.")


        

def main():
    cleaner = CafeSalesDataCleaner("./data/dirty_cafe_sales.csv")
    cleaner.load_data_csv()
    # initial_report = cleaner.generate_initial_report()
    # print(initial_report)

    # cleaner.clean_transaction_id()
    # cleaner.clean_item()
    # cleaner.clean_quantity()
    # cleaner.clean_price_per_unit()
    # cleaner.clean_total_spent()
    # cleaner.clean_payment_method()

    print(cleaner.df.iloc[10: 50])

if __name__ == "__main__":
    main()