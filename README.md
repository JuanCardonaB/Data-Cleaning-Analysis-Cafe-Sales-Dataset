## Café Sales Data: Cleaning, Processing & Analysis  

## 📌 Project Overview  
This project focuses on **data cleaning** and **exploratory data analysis (EDA)** using a simulated café sales dataset. The dataset was intentionally messy to practice real-world data wrangling skills, such as handling missing values, inconsistent entries, and incorrect data types.  

The main goals are:  
- Perform **data cleaning** with clear, rule-based transformations.  
- Conduct **exploratory analysis** to uncover insights about café sales.  
- Build a **reproducible workflow** that demonstrates professional data analysis practices.  

---

## 📂 Dataset Description  
The dataset (`dirty_cafe_sales.csv`) contains **10,000 transactions** with the following columns:  

| Column            | Description |
|-------------------|-------------|
| `Transaction ID`  | Unique identifier for each transaction. |
| `Item`            | Product purchased (e.g., Coffee, Sandwich, Cake). |
| `Quantity`        | Number of items purchased. |
| `Price Per Unit`  | Price of each unit of the product. |
| `Total Spent`     | Total amount spent in the transaction. |
| `Payment Method`  | Payment type (Cash, Credit Card, Digital Wallet, Unknown). |
| `Location`        | Transaction location (In-store, Takeaway, Unknown). |
| `Transaction Date`| Date of transaction. |

Dataset: https://www.kaggle.com/datasets/ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training

---

## 🧹 Data Cleaning Process  
To ensure a high-quality dataset, the following cleaning rules were applied:  

1. **Transaction ID**  
   - Kept as unique identifier.  

2. **Item**  
   - Missing values replaced with `"Unknown Item"`.  
   - `"UNKNOWN"` standardized to `"Unknown Item"`.  
   - Text normalized (consistent capitalization).  

3. **Quantity**  
   - Converted to integer.  
   - Missing values imputed with the **median**.  

4. **Price Per Unit**  
   - Converted to float.  
   - Missing values imputed with the **median per product** (`Item`).  

5. **Total Spent**  
   - Recalculated as `Quantity * Price Per Unit`.  
   - `"ERROR"` entries fixed by recomputing.  

6. **Payment Method**  
   - `"ERROR"` and `"UNKNOWN"` replaced with `"Unknown"`.  
   - Missing values replaced with `"Unknown"`.  

7. **Location**  
   - `"UNKNOWN"` replaced with `"Unknown"`.  
   - Missing values replaced with `"Unknown"`.  

8. **Transaction Date**  
   - Converted to `datetime`.  
   - Missing values imputed with the **median date**.  

---

## 📊 Exploratory Data Analysis (EDA)  
After cleaning, several analyses were performed:  

### 1. Descriptive Statistics  
- Total revenue and average revenue per transaction.  
- Distribution of `Quantity` and `Total Spent`.  

### 2. Sales by Product  
- Top-selling products by quantity.  
- Highest revenue-generating products.  

### 3. Payment Method Analysis  
- Most common payment methods.  
- Share of cash vs. digital payments.  

### 4. Location Analysis  
- Sales comparison between **In-store** and **Takeaway**.  

### 5. Time Series Analysis  
- Monthly sales trends.  
- Peak transaction days.  

---

## 📈 Key Insights  
- **Coffee** was the most frequently purchased product.  
- **Sandwiches** and **Cakes** generated significant revenue despite lower frequency.  
- **Credit Card** was the most used payment method, followed by **Cash**.  
- **In-store sales** dominated over Takeaway.  
- Clear **monthly seasonality** was observed, with spikes during weekends.  

---

## ⚙️ Tech Stack  
- **Python 3**  
- **pandas** (data manipulation)  
- **numpy** (numerical operations)  
- **matplotlib / seaborn** (visualizations)  
- **jupyter notebook** (analysis workflow)  

---

## 📁 Repository Structure  

cafe-sales-analysis/
│── data/
│ ├── dirty_cafe_sales.csv # Raw dataset
│ ├── clean_cafe_sales.csv # Cleaned dataset
│
│── notebooks/
│ ├── 01_data_cleaning.ipynb # Cleaning process
│ ├── 02_eda.ipynb # Exploratory data analysis
│
│── reports/
│ ├── sales_insights.pdf # Final analysis report
│
│── README.md # Project documentation

## 🚀 How to Run the Project  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/cafe-sales-analysis.git
   cd cafe-sales-analysis

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Open Jupyter Notebook and run the analysis:
   ```bash
   jupyter notebook

Author
Juan Cardona
📧 Contact: juanjocarbol@gmail.com
🔗 LinkedIn: (https://www.linkedin.com/in/juan-jose-cardona-bolivar/)
