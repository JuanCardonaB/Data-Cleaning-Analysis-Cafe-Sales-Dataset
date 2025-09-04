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
        # Rule: 
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
        # Rule: Must be positive integers
        # - Missing values → impute with median
        # - Convert to int
        pass


def main():
    cleaner = CafeSalesDataCleaner("./data/dirty_cafe_sales.csv")
    cleaner.load_data_csv()
    # initial_report = cleaner.generate_initial_report()
    # print(initial_report)

    # cleaner.clean_transaction_id()
    # cleaner.clean_item()

if __name__ == "__main__":
    main()