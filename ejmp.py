import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from pathlib import Path
import warnings

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suprimir warnings innecesarios
warnings.filterwarnings('ignore', category=FutureWarning)

class CafeSalesDataCleaner:
    """
    Clase profesional para la limpieza de datos de ventas de café.
    Implementa las reglas de negocio específicas para cada columna.
    """
    
    def __init__(self, filepath: str):
        """
        Inicializa el limpiador con la ruta del archivo.
        
        Args:
            filepath (str): Ruta al archivo CSV con datos sucios
        """
        self.filepath = Path(filepath)
        self.df = None
        self.cleaning_report = {}
        
        # Definir valores válidos para validaciones
        self.valid_payment_methods = {'Cash', 'Credit Card', 'Digital Wallet', 'Unknown'}
        self.valid_locations = {'In-store', 'Takeaway', 'Unknown'}
        
    def load_data(self) -> pd.DataFrame:
        """Carga los datos desde el archivo CSV."""
        try:
            self.df = pd.read_csv(self.filepath)
            logger.info(f"Datos cargados exitosamente. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error al cargar los datos: {e}")
            raise
    
    def generate_initial_report(self) -> Dict[str, Any]:
        """Genera un reporte inicial del estado de los datos."""
        report = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict(),
            'duplicates': self.df.duplicated().sum()
        }
        
        logger.info("Reporte inicial generado")
        return report
    
    def clean_transaction_id(self) -> None:
        """
        Limpia la columna Transaction ID.
        Regla: Sin nulos, mantener como identificador único.
        """
        initial_nulls = self.df['Transaction ID'].isnull().sum()
        
        if initial_nulls > 0:
            logger.warning(f"Se encontraron {initial_nulls} valores nulos en Transaction ID")
            # En un caso real, esto podría requerir investigación adicional
            self.df = self.df.dropna(subset=['Transaction ID'])
        
        # Verificar duplicados
        duplicates = self.df['Transaction ID'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Se encontraron {duplicates} Transaction IDs duplicados")
            self.df = self.df.drop_duplicates(subset=['Transaction ID'], keep='first')
        
        self.cleaning_report['transaction_id'] = {
            'nulls_removed': initial_nulls,
            'duplicates_removed': duplicates
        }
        
        logger.info("✅ Transaction ID limpiado")
    
    def clean_item(self) -> None:
        """
        Limpia la columna Item.
        Reglas: 
        - Valores faltantes → "Unknown Item"
        - "UNKNOWN" → "Unknown Item"
        - Tipificar valores (capitalización consistente)
        """
        initial_nulls = self.df['Item'].isnull().sum()
        
        # Reemplazar valores faltantes
        self.df['Item'] = self.df['Item'].fillna('Unknown Item')
        
        # Reemplazar "UNKNOWN"
        unknown_count = (self.df['Item'] == 'UNKNOWN').sum()
        self.df['Item'] = self.df['Item'].replace('UNKNOWN', 'Unknown Item')
        
        # Tipificar valores (capitalizar primera letra de cada palabra)
        self.df['Item'] = self.df['Item'].apply(
            lambda x: x.title() if isinstance(x, str) and x != 'Unknown Item' else x
        )
        
        self.cleaning_report['item'] = {
            'nulls_filled': initial_nulls,
            'unknowns_replaced': unknown_count,
            'unique_items': self.df['Item'].nunique()
        }
        
        logger.info(f"✅ Item limpiado - {initial_nulls} nulos y {unknown_count} 'UNKNOWN' reemplazados")
    
    def clean_quantity(self) -> None:
        """
        Limpia la columna Quantity.
        Reglas:
        - Valores faltantes → imputar con mediana
        - Convertir a int
        """
        initial_nulls = self.df['Quantity'].isnull().sum()
        
        # Calcular mediana excluyendo valores nulos
        median_quantity = self.df['Quantity'].median()
        
        # Imputar valores faltantes
        self.df['Quantity'] = self.df['Quantity'].fillna(median_quantity)
        
        # Convertir a int
        self.df['Quantity'] = self.df['Quantity'].astype(int)
        
        self.cleaning_report['quantity'] = {
            'nulls_filled': initial_nulls,
            'median_used': median_quantity
        }
        
        logger.info(f"✅ Quantity limpiado - {initial_nulls} valores imputados con mediana {median_quantity}")
    
    def clean_price_per_unit(self) -> None:
        """
        Limpia la columna Price Per Unit.
        Reglas:
        - Valores faltantes → imputar con mediana por producto
        - Convertir a float
        """
        initial_nulls = self.df['Price Per Unit'].isnull().sum()
        
        # Calcular mediana por Item
        median_by_item = self.df.groupby('Item')['Price Per Unit'].median()
        
        # Imputar valores faltantes usando la mediana por item
        def impute_price(row):
            if pd.isnull(row['Price Per Unit']):
                return median_by_item.get(row['Item'], self.df['Price Per Unit'].median())
            return row['Price Per Unit']
        
        self.df['Price Per Unit'] = self.df.apply(impute_price, axis=1)
        
        # Convertir a float
        self.df['Price Per Unit'] = self.df['Price Per Unit'].astype(float)
        
        self.cleaning_report['price_per_unit'] = {
            'nulls_filled': initial_nulls,
            'median_by_item_used': True
        }
        
        logger.info(f"✅ Price Per Unit limpiado - {initial_nulls} valores imputados por mediana de item")
    
    def clean_total_spent(self) -> None:
        """
        Limpia la columna Total Spent.
        Reglas:
        - Valores faltantes y "ERROR" → recalcular como Quantity * Price Per Unit
        """
        initial_nulls = self.df['Total Spent'].isnull().sum()
        error_count = (self.df['Total Spent'] == 'ERROR').sum() if 'Total Spent' in self.df.columns else 0
        
        # Reemplazar "ERROR" con NaN
        self.df['Total Spent'] = self.df['Total Spent'].replace('ERROR', np.nan)
        
        # Recalcular todos los valores como Quantity * Price Per Unit
        self.df['Total Spent'] = self.df['Quantity'] * self.df['Price Per Unit']
        
        self.cleaning_report['total_spent'] = {
            'initial_nulls': initial_nulls,
            'errors_replaced': error_count,
            'recalculated_all': True
        }
        
        logger.info(f"✅ Total Spent recalculado completamente - {initial_nulls + error_count} valores corregidos")
    
    def clean_payment_method(self) -> None:
        """
        Limpia la columna Payment Method.
        Reglas:
        - Valores faltantes → "Unknown"
        - "ERROR", "UNKNOWN" → "Unknown"
        - Validar categorías válidas
        """
        initial_nulls = self.df['Payment Method'].isnull().sum()
        
        # Reemplazar valores faltantes
        self.df['Payment Method'] = self.df['Payment Method'].fillna('Unknown')
        
        # Reemplazar valores problemáticos
        error_unknown_count = (
            (self.df['Payment Method'] == 'ERROR') | 
            (self.df['Payment Method'] == 'UNKNOWN')
        ).sum()
        
        self.df['Payment Method'] = self.df['Payment Method'].replace({
            'ERROR': 'Unknown',
            'UNKNOWN': 'Unknown'
        })
        
        # Validar que solo existan categorías válidas
        invalid_methods = set(self.df['Payment Method'].unique()) - self.valid_payment_methods
        if invalid_methods:
            logger.warning(f"Métodos de pago inválidos encontrados: {invalid_methods}")
            self.df.loc[self.df['Payment Method'].isin(invalid_methods), 'Payment Method'] = 'Unknown'
        
        self.cleaning_report['payment_method'] = {
            'nulls_filled': initial_nulls,
            'errors_replaced': error_unknown_count,
            'invalid_methods_found': len(invalid_methods)
        }
        
        logger.info(f"✅ Payment Method limpiado - {initial_nulls} nulos y {error_unknown_count} errores reemplazados")
    
    def clean_location(self) -> None:
        """
        Limpia la columna Location.
        Reglas:
        - Valores faltantes → "Unknown"
        - "UNKNOWN" → "Unknown"
        - Validar categorías válidas
        """
        initial_nulls = self.df['Location'].isnull().sum()
        
        # Reemplazar valores faltantes
        self.df['Location'] = self.df['Location'].fillna('Unknown')
        
        # Reemplazar "UNKNOWN"
        unknown_count = (self.df['Location'] == 'UNKNOWN').sum()
        self.df['Location'] = self.df['Location'].replace('UNKNOWN', 'Unknown')
        
        # Validar que solo existan categorías válidas
        invalid_locations = set(self.df['Location'].unique()) - self.valid_locations
        if invalid_locations:
            logger.warning(f"Ubicaciones inválidas encontradas: {invalid_locations}")
            self.df.loc[self.df['Location'].isin(invalid_locations), 'Location'] = 'Unknown'
        
        self.cleaning_report['location'] = {
            'nulls_filled': initial_nulls,
            'unknowns_replaced': unknown_count,
            'invalid_locations_found': len(invalid_locations)
        }
        
        logger.info(f"✅ Location limpiado - {initial_nulls} nulos y {unknown_count} 'UNKNOWN' reemplazados")
    
    def clean_transaction_date(self) -> None:
        """
        Limpia la columna Transaction Date.
        Reglas:
        - Valores faltantes → imputar con mediana de fechas
        - Convertir a datetime
        """
        initial_nulls = self.df['Transaction Date'].isnull().sum()
        
        # Convertir a datetime primero
        self.df['Transaction Date'] = pd.to_datetime(
            self.df['Transaction Date'], 
            errors='coerce'
        )
        
        # Calcular mediana de fechas válidas
        valid_dates = self.df['Transaction Date'].dropna()
        if len(valid_dates) > 0:
            median_date = valid_dates.median()
            # Imputar valores faltantes con la mediana
            self.df['Transaction Date'] = self.df['Transaction Date'].fillna(median_date)
        
        self.cleaning_report['transaction_date'] = {
            'nulls_filled': initial_nulls,
            'median_date_used': median_date if len(valid_dates) > 0 else None
        }
        
        logger.info(f"✅ Transaction Date limpiado - {initial_nulls} valores imputados")
    
    def validate_cleaned_data(self) -> Dict[str, Any]:
        """Valida que los datos limpiados cumplan con las reglas de negocio."""
        validation_report = {}
        
        # Validar Transaction ID
        validation_report['transaction_id_unique'] = self.df['Transaction ID'].is_unique
        validation_report['transaction_id_nulls'] = self.df['Transaction ID'].isnull().sum()
        
        # Validar tipos de datos
        validation_report['quantity_is_int'] = self.df['Quantity'].dtype == 'int64'
        validation_report['price_is_float'] = self.df['Price Per Unit'].dtype == 'float64'
        validation_report['date_is_datetime'] = pd.api.types.is_datetime64_any_dtype(self.df['Transaction Date'])
        
        # Validar categorías
        validation_report['valid_payment_methods'] = set(self.df['Payment Method'].unique()).issubset(self.valid_payment_methods)
        validation_report['valid_locations'] = set(self.df['Location'].unique()).issubset(self.valid_locations)
        
        # Validar Total Spent
        calculated_total = self.df['Quantity'] * self.df['Price Per Unit']
        validation_report['total_spent_correct'] = np.allclose(self.df['Total Spent'], calculated_total, rtol=1e-10)
        
        return validation_report
    
    def clean_all(self) -> pd.DataFrame:
        """
        Ejecuta todo el proceso de limpieza en el orden correcto.
        
        Returns:
            pd.DataFrame: DataFrame limpio
        """
        logger.info("🚀 Iniciando proceso de limpieza de datos")
        
        # Cargar datos
        self.load_data()
        
        # Generar reporte inicial
        initial_report = self.generate_initial_report()
        logger.info(f"Estado inicial: {initial_report['total_rows']} filas, {initial_report['total_columns']} columnas")
        
        # Ejecutar limpieza en orden
        self.clean_transaction_id()
        self.clean_item()
        self.clean_quantity()
        self.clean_price_per_unit()
        self.clean_total_spent()
        self.clean_payment_method()
        self.clean_location()
        self.clean_transaction_date()
        
        # Validar datos limpiados
        validation_report = self.validate_cleaned_data()
        
        # Compilar reporte final
        self.cleaning_report['initial_state'] = initial_report
        self.cleaning_report['validation'] = validation_report
        self.cleaning_report['final_shape'] = self.df.shape
        
        logger.info("✅ Proceso de limpieza completado exitosamente")
        
        return self.df
    
    def save_cleaned_data(self, output_path: str) -> None:
        """Guarda los datos limpiados en un archivo CSV."""
        output_path = Path(output_path)
        self.df.to_csv(output_path, index=False)
        logger.info(f"Datos limpiados guardados en: {output_path}")
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Retorna el reporte completo de limpieza."""
        return self.cleaning_report


def main():
    """Función principal para ejecutar la limpieza de datos."""
    
    # Configurar rutas
    input_file = "../data/dirty_cafe_sales.csv"
    output_file = "../data/clean_cafe_sales.csv"
    
    try:
        # Inicializar limpiador
        cleaner = CafeSalesDataCleaner(input_file)
        
        # Ejecutar limpieza
        clean_df = cleaner.clean_all()
        
        # Guardar datos limpiados
        cleaner.save_cleaned_data(output_file)
        
        # Mostrar reporte final
        report = cleaner.get_cleaning_report()
        
        print("\n" + "="*60)
        print("📊 REPORTE FINAL DE LIMPIEZA")
        print("="*60)
        
        print(f"📈 Filas procesadas: {report['final_shape'][0]}")
        print(f"📋 Columnas: {report['final_shape'][1]}")
        
        print("\n🔧 Acciones realizadas por columna:")
        for column, actions in report.items():
            if column not in ['initial_state', 'validation', 'final_shape']:
                print(f"  • {column}: {actions}")
        
        print("\n✅ Validaciones:")
        for validation, result in report['validation'].items():
            status = "✅" if result else "❌"
            print(f"  {status} {validation}: {result}")
        
        print(f"\n💾 Datos limpiados guardados en: {output_file}")
        
    except Exception as e:
        logger.error(f"Error en el proceso de limpieza: {e}")
        raise


if __name__ == "__main__":
    main()