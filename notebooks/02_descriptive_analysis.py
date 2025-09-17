import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
from pathlib import Path
from typing import Dict, Any

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)

class CafeSalesDescriptiveAnalysis:
    # Class for performing descriptive analysis of cafe sales data
    # Analyzes basic patterns, top-selling products, payment methods, etc.
    def __init__(self, filepath):
        self.file_path = Path(filepath)
        self.df = None
        self.analysis_results = {}

    def load_data(self) -> pd.DataFrame:
        # Load the cafe sales data from a CSV file
        try:
            self.df = pd.read_csv(self.file_path)
            logger.info("Data loaded successfully.")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def basic_statistics(self) -> Dict[str, Any]:
        # Calculate basic sales statistics
        # Returns:
        #   - Dict: Dictionary with key statistics

        # TOTAL SALES CALCULATIONS
        total_sales = self.df['Total Spent'].sum()
        total_transactions = len(self.df)
        avg_transaction = self.df['Total Spent'].mean()

        # PRODUCT CALCULATIONS
        total_items_sold = self.df['Quantity'].sum()
        avg_items_per_transaction = self.df['Quantity'].mean()

        # PRICE CALCULATIONS
        avg_price_per_unit = self.df['Price Per Unit'].mean()

        stats = {
            'total_sales': total_sales,
            'total_transactions': total_transactions,
            'avg_transaction': avg_transaction,
            'total_items_sold': total_items_sold,
            'avg_items_per_transaction': avg_items_per_transaction,
            'avg_price_per_unit': avg_price_per_unit
        }

        self.analysis_results['basic_stats'] = stats
        logger.info("Basic statistics calculated.")
        return stats

    def top_selling_products(self, top_n: int = 10) -> Dict[str, pd.DataFrame]:
        # Identify top-selling products by quantity and revenue
        # 
        # Args:
        #     top_n (int): Number of top products to show    
        # Returns:
        #     Dict: Dictionary with DataFrames of top products

        # TOP PRODUCTS BY QUANTITY
        products_by_quantity = (
            self.df.groupby('Item')['Quantity']
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

        # TOP PRODUCTS BY REVENUE
        products_by_revenue = (
            self.df.groupby('Item')['Total Spent']
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

        top_products = {
            'by_quantity': products_by_quantity,
            'by_revenue': products_by_revenue
        }

        self.analysis_results['top_products'] = top_products
        logger.info("Top-selling products identified.")
        return top_products
    
    def payment_method_analysis(self) -> Dict[str, Any]:
        # Analyze payment method distribution
        # Returns:
        #   Dict: Payment method statistics 

        # TRANSACTION COUNT BY METHOD
        payment_counts = self.df['Payment Method'].value_counts()

        # REVENUE BY PAYMENT METHOD
        payments_revenue = (
            self.df.groupby('Payment Method')['Total Spent']
            .sum()
            .sort_values(ascending=False)
        )

        # AVERAGE TICKET BY METHOD
        payment_avg_ticket = (
            self.df.groupby('Payment Method')['Total Spent']
            .mean()
            .sort_values(ascending=False)
        )

        # STORE RESULTS
        payments_analysis = {
            'transaction_counts': payment_counts,
            'payments_revenue': payments_revenue,
            'payment_avg_ticket': payment_avg_ticket
        }

        self.analysis_results['payment_methods'] = payments_analysis
        logger.info("Payment method analysis completed.")
        return payments_analysis

if __name__ == "__main__":
    data_path = './data/processed/cleaned_cafe_sales.csv'
    analysis = CafeSalesDescriptiveAnalysis(data_path)
    df = analysis.load_data()
    # stats = analysis.basic_statistics()
    # print(stats)

    # products = analysis.top_selling_products(top_n=5)
    # print(products["by_quantity"])
    # print()
    # print(products["by_revenue"])

    payments = analysis.payment_method_analysis()
    print(payments["transaction_counts"])
    print()
    print(payments["payments_revenue"])
    print()
    print(payments["payment_avg_ticket"])