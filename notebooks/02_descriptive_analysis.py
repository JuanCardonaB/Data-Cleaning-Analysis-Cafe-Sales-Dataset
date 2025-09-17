import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
from pathlib import Path
from typing import Dict, Any
import numpy as np
import json

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
    """Class for performing descriptive analysis of cafe sales data"""
    def __init__(self, filepath):
        self.file_path = Path(filepath)
        self.df = None
        self.analysis_results = {}

    def load_data(self) -> pd.DataFrame:
        """Load the cafe sales data from a CSV file
        
        Returns:
            pd.DataFrame: DataFrame with loaded data
        """
        try:
            self.df = pd.read_csv(self.file_path)
            logger.info("Data loaded successfully.")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def basic_statistics(self) -> Dict[str, Any]:
        """Calculate basic sales statistics
        
        Returns:
            Dict: Dictionary with key statistics
        """
        total_sales = self.df['Total Spent'].sum()
        total_transactions = len(self.df)
        avg_transaction = self.df['Total Spent'].mean()
        total_items_sold = self.df['Quantity'].sum()
        avg_items_per_transaction = self.df['Quantity'].mean()
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
        """Identify top-selling products by quantity and revenue
        
        Args:
            top_n (int): Number of top products to show
            
        Returns:
            Dict: Dictionary with DataFrames of top products
        """
        products_by_quantity = (
            self.df.groupby('Item')['Quantity']
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )

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
        """Analyze payment method distribution
        
        Returns:
            Dict: Payment method statistics
        """
        payment_counts = self.df['Payment Method'].value_counts()

        payments_revenue = (
            self.df.groupby('Payment Method')['Total Spent']
            .sum()
            .sort_values(ascending=False)
        )

        payment_avg_ticket = (
            self.df.groupby('Payment Method')['Total Spent']
            .mean()
            .sort_values(ascending=False)
        )

        payments_analysis = {
            'transaction_counts': payment_counts.to_dict(),
            'payments_revenue': payments_revenue.to_dict(),
            'payment_avg_ticket': payment_avg_ticket.to_dict()
        }

        self.analysis_results['payment_methods'] = payments_analysis
        logger.info("Payment method analysis completed.")
        return payments_analysis

    def location_analysis(self) -> Dict[str, Any]:
        """Analyze sales distribution by location
        
        Returns:
            Dict: Location statistics
        """
        location_counts = self.df['Location'].value_counts()

        location_revenue = (
            self.df.groupby('Location')['Total Spent']
            .sum()
            .sort_values(ascending=False)
        )

        location_avg_ticket = (
            self.df.groupby('Location')['Total Spent']
            .mean()
            .sort_values(ascending=False)
        )

        location_items = (
            self.df.groupby('Location')['Quantity']
            .sum()
            .sort_values(ascending=False)
        )

        location_analysis = {
            'transaction_counts': location_counts.to_dict(),
            'revenue_by_location': location_revenue.to_dict(),
            'avg_ticket_by_location': location_avg_ticket.to_dict(),
            'items_sold_by_location': location_items.to_dict()
        }

        self.analysis_results['location_analysis'] = location_analysis
        logger.info("Location analysis completed.")
        return location_analysis

    def create_visualizations(self):
        """Create visualizations for the analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cafe Sales Descriptive Analysis', fontsize=16, fontweight='bold')

        top_products_qty = self.analysis_results['top_products']['by_quantity'].head(10)
        axes[0, 0].barh(top_products_qty['Item'], top_products_qty['Quantity'])
        axes[0, 0].set_title('Top 10 Products by Quantity Sold')
        axes[0, 0].set_xlabel('Quantity')

        payment_data = self.analysis_results['payment_methods']['payments_revenue']
        axes[0, 1].pie(payment_data.values(), labels=payment_data.keys(), autopct='%1.1f%%')
        axes[0, 1].set_title('Revenue Distribution by Payment Method')

        location_data = self.analysis_results['location_analysis']['transaction_counts']
        axes[1, 0].bar(location_data.keys(), location_data.values())
        axes[1, 0].set_title('Transactions by Location')
        axes[1, 0].set_ylabel('Number of Transactions')

        avg_ticket_data = self.analysis_results['payment_methods']['payment_avg_ticket']
        axes[1, 1].bar(avg_ticket_data.keys(), avg_ticket_data.values())
        axes[1, 1].set_title('Average Ticket by Payment Method')
        axes[1, 1].set_ylabel('Average Ticket ($)')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

        logger.info("Visualizations created.")

    def generate_summary_report(self) -> str:
        """Generate a text summary report
        
        Returns:
            str: Formatted report
        """
        report = "="*50 + "\n"
        report += "   DESCRIPTIVE ANALYSIS REPORT\n"
        report += "="*50 + "\n\n"
        
        stats = self.analysis_results['basic_stats']
        report += "GENERAL STATISTICS:\n"
        report += f"• Total sales: ${stats['total_sales']:,.2f}\n"
        report += f"• Total transactions: {stats['total_transactions']:,}\n"
        report += f"• Average ticket: ${stats['avg_transaction']:.2f}\n"
        report += f"• Products sold: {stats['total_items_sold']:,}\n\n"
        
        top_product = self.analysis_results['top_products']['by_quantity'].iloc[0]
        report += "STAR PRODUCT (by quantity):\n"
        report += f"• {top_product['Item']}: {top_product['Quantity']} units\n\n"
        
        top_payment = max(self.analysis_results['payment_methods']['transaction_counts'].items(),
                         key=lambda x: x[1])
        report += "MOST USED PAYMENT METHOD:\n"
        report += f"• {top_payment[0]}: {top_payment[1]} transactions\n\n"
        
        return report

    def save_results(self, report_path: str) -> None:
        """Save analysis results to JSON file
        
        Args:
            report_path (str): Path to save JSON file
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
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict('records')
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
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
    
            clean_report = convert_numpy_types(self.analysis_results)
    
            with open(report_path, 'w') as f:
                json.dump(clean_report, f, indent=4, ensure_ascii=False)
    
            logger.info(f"Analysis results saved to {report_path}")
    
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
            raise
    
    def run_complete_analysis(self, output_path_report) -> Dict[str, Any]:
        """Execute complete descriptive analysis
        
        Returns:
            Dict: All analysis results
        """
        logger.info("Starting complete descriptive analysis")

        self.load_data()

        self.basic_statistics()
        self.top_selling_products()
        self.payment_method_analysis()
        self.location_analysis()

        report = self.generate_summary_report()
        print(report)

        self.save_results(output_path_report)

        self.create_visualizations()

        logger.info("Descriptive analysis completed")
        return self.analysis_results

if __name__ == "__main__":
    data_path = './data/processed/cleaned_cafe_sales.csv'
    analysis = CafeSalesDescriptiveAnalysis(data_path)
    results = analysis.run_complete_analysis('./reports/descriptive_analysis_report.json')
    