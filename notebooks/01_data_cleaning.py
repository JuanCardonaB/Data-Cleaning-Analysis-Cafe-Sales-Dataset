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

class CafeSalesDataCleaner:
    """Class for cleaning cafe sales data"""
    def __init__(self, filepath: str):
        self.file_path = Path(filepath)
        self.df = None
        self.cleaning_report = {}

        self.valid_payment_methods = {'Cash', 'Credit Card', 'Digital Wallet', 'Unknown'}
        self.valid_locations = {'In-Store', 'Takeaway', 'Unknown'}

    def load_data_csv(self) -> pd.DataFrame:
        """Load data from CSV file
        
        Returns:
            pd.DataFrame: DataFrame with loaded data
        """
        try:
            self.df = pd.read_csv(self.file_path)
            logger.info(f"Data loaded successfully from {self.file_path}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def generate_initial_report(self) -> Dict[str, Any]:
        """Generate initial data quality report
        
        Returns:
            Dict: Report with data statistics
        """
        report = {
            "num_rows": int(len(self.df)),
            "num_columns": int(len(self.df.columns)),
            "missing_values": {k: int(v) for k, v in self.df.isnull().sum().to_dict().items()},
            "data_types": {k: str(v) for k, v in self.df.dtypes.to_dict().items()},
            "duplicates": int(self.df.duplicated().sum())
        }

        logger.info("Initial data report generated")
        return report
    
    def clean_transaction_id(self) -> None:
        """Clean the Transaction ID column
        
        Rules: Must be unique and non-null
        """
        initial_null_count = self.df['Transaction ID'].isnull().sum()

        if initial_null_count > 0:
            logger.info(f"Found {initial_null_count} null Transaction IDs.")
            self.df = self.df.dropna(subset=['Transaction ID'])
        
        duplicates = self.df['Transaction ID'].duplicated().sum()
        if duplicates > 0:
            logger.info(f"Found {duplicates} duplicate Transaction IDs. Removing duplicates.")
            self.df = self.df.drop_duplicates(subset=['Transaction ID'])

        self.cleaning_report["Transaction ID"] = {
            "nulls_removed": initial_null_count,
            "duplicates_removed": duplicates
        }

        logger.info("Transaction ID column cleaned.")

    def clean_item(self) -> None:
        """Clean the Item column
        
        Rules: 
        - Missing values → "Unknown Item"
        - "UNKNOWN" → "Unknown Item"
        - Standardize values (consistent capitalization)
        """
        initial_null_count = self.df['Item'].isnull().sum()
        logger.info(f"Found {initial_null_count} null Item entries.")

        self.df['Item'] = self.df['Item'].fillna('Unknown Item')

        unknown_count = (self.df['Item'].str.upper() == 'UNKNOWN').sum()
        self.df['Item'] = self.df['Item'].str.upper().replace('UNKNOWN', 'Unknown Item')

        self.df['Item'] = self.df['Item'].apply(
            lambda x: x.title() if isinstance(x, str) and x != 'Unknown Item' else x
        )

        self.cleaning_report["Item"] = {
            'nulls_filled': initial_null_count,
            'unknown_replaced': unknown_count,
            'unique_items': self.df['Item'].nunique()
        }

        logger.info("Item column cleaned.")

    def clean_quantity(self) -> None:
        """Clean the Quantity column
        
        Rules:
        - Missing values → impute with median
        - Convert to int
        """
        initial_null_count = self.df['Quantity'].isnull().sum()

        self.df['Quantity'] = pd.to_numeric(self.df['Quantity'], errors='coerce')

        median_quantity = self.df['Quantity'].median()

        self.df['Quantity'] = self.df['Quantity'].fillna(median_quantity)

        self.df['Quantity'] = self.df['Quantity'].astype(int)

        self.cleaning_report["Quantity"] = {
            'nulls_filled': initial_null_count,
            'median_used': median_quantity
        }

        logger.info("Quantity column cleaned.")

    def clean_price_per_unit(self) -> None:
        """Clean the Price Per Unit column
        
        Rules:
        - Missing values → impute with median per product
        - Convert to float
        """
        initial_null_count = self.df['Price Per Unit'].isnull().sum()

        self.df['Price Per Unit'] = pd.to_numeric(self.df['Price Per Unit'], errors='coerce')

        median_by_item = self.df.groupby('Item')['Price Per Unit'].median()

        def impute_price(row):
            if pd.isnull(row['Price Per Unit']):
                return median_by_item.get(row['Item'], self.df['Price Per Unit'].median())
            return row['Price Per Unit']
        
        self.df['Price Per Unit'] = self.df.apply(impute_price, axis=1)

        self.df['Price Per Unit'] = self.df['Price Per Unit'].astype(float)

        self.cleaning_report["Price Per Unit"] = {
            'nulls_filled': initial_null_count,
             'median_by_item_used': True
        }

        logger.info("Price Per Unit column cleaned.")

    def clean_total_spent(self) -> None:
        """Clean the Total Spent column
        
        Rules:
        - Missing values and "ERROR" → recalculate as Quantity * Price Per Unit
        """
        initial_null_count = self.df['Total Spent'].isnull().sum()
        error_count = (self.df['Total Spent'] == 'ERROR').sum() if 'Total Spent' in self.df.columns else 0

        self.df['Total Spent'] = self.df['Quantity'] * self.df['Price Per Unit']

        self.cleaning_report["Total Spent"] = {
            'initial_nulls': initial_null_count,
            'errors_replaced': error_count,
            'recalculated_all': True
        }

        logger.info("Total Spent column cleaned.")

    def clean_payment_method(self) -> None:
        """Clean the Payment Method column
        
        Rules:
        - Missing values and "ERROR" → "Unknown"
        """
        initial_null_count = self.df['Payment Method'].isnull().sum()
        error_count = (
            (self.df['Payment Method'].str.upper() == 'ERROR') |
            (self.df['Payment Method'].str.upper() == 'UNKNOWN')
        ).sum()

        self.df['Payment Method'] = self.df['Payment Method'].fillna('Unknown')

        self.df['Payment Method'] = self.df['Payment Method'].str.upper().replace({
            'ERROR': 'Unknown',
            'UNKNOWN': 'Unknown'
        })

        self.df['Payment Method'] = self.df['Payment Method'].apply(
            lambda x: x.title() if isinstance(x, str) and x != 'Unknown' else x
        )

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

    def clean_location(self) -> None:
        """Clean the Location column
        
        Rules:
        - Missing values and "UNKNOWN" → "Unknown"
        - Standardize capitalization
        - Validate valid categories
        """
        initial_null_count = self.df['Location'].isnull().sum()

        self.df['Location'] = self.df['Location'].fillna('Unknown')

        unknown_count = (self.df['Location'].str.upper() == 'UNKNOWN').sum()
        self.df['Location'] = self.df['Location'].str.upper().replace({
            'UNKNOWN': 'Unknown',
            'ERROR': 'Unknown'
        })

        self.df['Location'] = self.df['Location'].apply(
            lambda x: x.title() if isinstance(x, str) and x != 'Unknown' else x
        )

        invalid_locations = set(self.df['Location'].unique()) - self.valid_locations
        if invalid_locations:
            logger.warning(f"Found invalid location categories: {invalid_locations}. Replacing with 'Unknown'.")
            self.df.loc[self.df['Location'].isin(invalid_locations), 'Location'] = 'Unknown'

        self.cleaning_report["Location"] = {
            'nulls_filled': initial_null_count,
            'errors_replaced': unknown_count,
            'invalid_locations_replaced': len(invalid_locations)
        }

        logger.info("Location column cleaned.")

    def clean_transaction_date(self) -> None:
        """Clean the Transaction Date column
        
        Rules:
        - Missing values → impute with mode
        - Convert to datetime
        """
        initial_nulls = self.df['Transaction Date'].isnull().sum()

        self.df['Transaction Date'] = pd.to_datetime(
            self.df['Transaction Date'], 
            errors='coerce'
        )

        valid_dates = self.df['Transaction Date'].dropna()
        if len(valid_dates) > 0:
            median_date = valid_dates.median()
            self.df['Transaction Date'] = self.df['Transaction Date'].fillna(median_date)

        self.cleaning_report["Transaction Date"] = {
            'nulls_filled': initial_nulls,
            'median_used': median_date if len(valid_dates) > 0 else None
        }

        logger.info("Transaction Date column cleaned.")

    def validate_cleaned_data(self) -> Dict[str, Any]:
        """Validate the cleaned data quality
        
        Returns:
            Dict: Validation report
        """
        validation_report = {}

        validation_report['transaction_id_unique'] = self.df['Transaction ID'].is_unique
        validation_report['transaction_id_nulls'] = self.df['Transaction ID'].isnull().sum()

        validation_report['quantity_is_int'] = self.df['Quantity'].dtype == 'int64'
        validation_report['price_is_float'] = self.df['Price Per Unit'].dtype == 'float64'
        validation_report['transaction_date_is_datetime'] = pd.api.types.is_datetime64_any_dtype(self.df['Transaction Date'])

        validation_report['valid_payment_methods'] = set(self.df['Payment Method'].unique()).issubset(self.valid_payment_methods)
        validation_report['valid_locations'] = set(self.df['Location'].unique()).issubset(self.valid_locations)

        calculated_total = self.df['Quantity'] * self.df['Price Per Unit']
        validation_report['total_spent_correct'] = np.allclose(self.df['Total Spent'], calculated_total, rtol=1e-10)
        
        return validation_report

    def clean_all(self) -> None:
        """Run all cleaning steps
        
        Returns:
            pd.DataFrame: The cleaned dataframe
        """
        logger.info("Starting full data cleaning process.")

        self.load_data_csv()

        initial_report = self.generate_initial_report()
        logger.info(f"Initial Data Report: {initial_report['num_rows']} rows, {initial_report['num_columns']} columns")

        self.clean_transaction_id()
        self.clean_item()
        self.clean_quantity()
        self.clean_price_per_unit()
        self.clean_total_spent()
        self.clean_payment_method()
        self.clean_location()
        self.clean_transaction_date()   

        validation_report = self.validate_cleaned_data()

        self.cleaning_report['initial_state'] = initial_report
        self.cleaning_report['validation'] = validation_report
        self.cleaning_report['final_state'] = self.df.shape

        logger.info("Data cleaning process completed.")

        return self.df

    def save_cleaned_data(self, output_path: str) -> None:
        """Save the cleaned dataframe to a CSV file
        
        Args:
            output_path (str): Path to save the cleaned CSV file
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self.df.to_csv(output_path, index=False)
            logger.info(f"Cleaned data saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving cleaned data: {e}")
            raise
    
    def save_cleaning_report(self, report_path: str) -> None:
        """Save the cleaning report to a JSON file
        
        Args:
            report_path (str): Path to save the cleaning report
        """
        try:
            import json
            report_path = Path(report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
    
            def convert_numpy_types(obj):
                """Convert numpy/pandas types to Python native types for JSON"""
                if hasattr(obj, 'item'):
                    return obj.item()
                elif hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_numpy_types(item) for item in obj)
                elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif pd.isna(obj):
                    return None
                elif hasattr(obj, 'name'):
                    return str(obj)
                elif hasattr(obj, '__str__'):
                    return str(obj)
                else:
                    return obj
    
            clean_report = convert_numpy_types(self.cleaning_report)
    
            with open(report_path, 'w') as f:
                json.dump(clean_report, f, indent=4, ensure_ascii=False)
    
            logger.info(f"Cleaning report saved to {report_path}")
    
        except Exception as e:
            logger.error(f"Error saving cleaning report: {e}")
            raise

def main():
    """Main function to execute the data cleaning workflow"""
    input_file = "./data/raw/dirty_cafe_sales.csv"
    output_file = "./data/processed/cleaned_cafe_sales.csv"
    report_file = "./reports/cleaning_report.json"

    try:
        cleaner = CafeSalesDataCleaner(input_file)

        cleaned_df = cleaner.clean_all()

        cleaner.save_cleaned_data(output_file)

        cleaner.save_cleaning_report(report_file)
        logger.info("\n\n\nData cleaning workflow completed successfully.\n")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return


if __name__ == "__main__":
    main()