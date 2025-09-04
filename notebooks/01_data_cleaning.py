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
        initial_null_count = self.df['Transaction ID'].isnull().sum()

        if initial_null_count > 0:
            logger.info(f"Found {initial_null_count} null Transaction IDs.")
            self.df = self.df.dropna(subset=['Transaction ID']) # Remove rows with null Transaction ID
        
        # Verify duplicates
        duplicates = self.df['Transaction ID'].duplicated().sum()
        logger.info(f"Found {duplicates} duplicate Transaction IDs.")

def main():
    cleaner = CafeSalesDataCleaner("./data/dirty_cafe_sales.csv")
    cleaner.load_data_csv()
    # initial_report = cleaner.generate_initial_report()
    # print(initial_report)
    cleaner.clean_transaction_id()

if __name__ == "__main__":
    main()