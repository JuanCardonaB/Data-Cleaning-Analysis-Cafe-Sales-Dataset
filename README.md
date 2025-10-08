# ☕ Café Sales Data Cleaning Project  

## 📌 Project Overview  
This project focuses **exclusively on data cleaning** using a simulated café sales dataset from Kaggle.  
The goal is to practice real-world **data wrangling** techniques by transforming a messy dataset into a clean, reliable, and analysis-ready version.  

---

## 🧹 Data Cleaning Objectives  
The dataset (`dirty_cafe_sales.csv`) contains **10,000 transactions** with common data issues such as missing values, inconsistent text, and incorrect data types.  
The cleaning process aims to:  

- Handle **missing and inconsistent values**.  
- Fix **incorrect data types**.  
- Standardize text and categorical fields.  
- Recalculate fields with logical errors.  
- Produce a final **clean dataset** ready for analysis.  

---

## ⚙️ Cleaning Rules Applied  

| Column | Cleaning Actions |
|--------|------------------|
| **Transaction ID** | Kept as unique identifier. |
| **Item** | Missing values replaced with `"Unknown Item"`, standardized text. |
| **Quantity** | Converted to integer, missing values imputed with median. |
| **Price Per Unit** | Converted to float, missing values imputed with median per product. |
| **Total Spent** | Recalculated as `Quantity * Price Per Unit`. |
| **Payment Method** | Standardized values and replaced unknowns with `"Unknown"`. |
| **Location** | Standardized and filled missing values with `"Unknown"`. |
| **Transaction Date** | Converted to datetime and missing values imputed with median date. |

---

## 🧾 Output  
After cleaning, the resulting dataset is saved as:  
`data/processed/cleaned_cafe_sales.csv`

## ⚙️ Tech Stack  
- **Python 3**  
- **pandas** (data manipulation)  
- **numpy** (numerical operations)  
- **jupyter notebook** (cleaning workflow)  

---

## 🚀 How to Run the Project  

1. Clone the repository:
   
   ```git clone https://github.com/JuanCardonaB/Data-Cleaning-Analysis-Caf-Sales-Dataset.git```
   
   ``` cd Data-Cleaning-Analysis-Caf-Sales-Dataset```

2. Install dependencies:

   ```pip install -r requirements.txt```


---

Author: Juan Cardona

📧 Contact: juanjocarbol@gmail.com

🔗 LinkedIn: linkedin.com/in/juan-jose-cardona-bolivar

