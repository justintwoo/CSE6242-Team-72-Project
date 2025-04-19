# 🏡 CSE 6242 Project - Housing Price Prediction (Spring 2025)

## Project Title
**Predicting Housing Prices in the US with Explainable AI: A Machine Learning Approach**

---

## Group Members
- Justin Woo  
- Pegah Mirghafari  
- Hussain Ather  
- Wei Liu  
- Minhall Shen  

---

## 📌 Project Summary
This project uses machine learning and explainable AI to model and predict housing prices in the United States.  
Our goal is to make accurate, interpretable predictions by integrating geographic, structural, and socioeconomic data.

We build multiple models and compare their performance, then use **SHAP (Shapley Additive Explanations)** to visualize and explain which features impact prices the most.

We also provide a **Streamlit-based interactive dashboard** that allows users to explore the dataset visually by state, price, and other attributes.

---

## 📁 Repository Structure

```
CSE6242-Team-72-Project/
│
├── EDA.ipynb              --> Data cleaning & exploratory analysis
├── main.py                --> Model training & SHAP explainability (script)
├── main.ipynb             --> Same as above, in Jupyter Notebook form
├── requirements.txt       --> Python dependencies
├── README.md              --> Project documentation (this file)
│
├── app/
│   └── app.py             --> Streamlit dashboard (UI)
│
└── data/
    └── realtor-data.csv   --> Kaggle dataset (NOT included)
```

---

## 📦 Dataset Instructions

**Dataset Source:**  
[USA Real Estate Dataset on Kaggle](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset)

### 🔻 How to Download the Dataset:
1. Visit the Kaggle dataset page:  
   https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset
2. Click **"Download"**
3. Extract the ZIP file
4. Move `realtor-data.csv` to:
   ```
   CSE6242-Team-72-Project/data/realtor-data.csv
   ```

Make sure the final CSV file is located at:
```
CSE6242-Team-72-Project/data/realtor-data.csv
```

---

## 🧮 Features Used

| Feature         | Description                      |
|----------------|----------------------------------|
| `bed`           | Number of bedrooms               |
| `bath`          | Number of bathrooms              |
| `acre_lot`      | Lot size in acres                |
| `house_size`    | House square footage             |
| `city_encoded`  | Average price by city            |
| `state_encoded` | Average price by state           |

---

## 🧠 Modeling Summary

- **Best Model:** Random Forest  
- **R² Score:** 0.9612  
- **RMSE:** $36,138  
- **MAE:** $24,369  

Other models evaluated: XGBoost, Linear Regression  
SHAP was used to explain feature importance across models.

---

## 🚀 Running the Dashboard

To launch the interactive dashboard:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```

**Dashboard features:**
- Interactive choropleth map
- Price distribution histograms
- Filter by state & price
- Optional SHAP visualizations

---

## 🛠️ Tech Stack

- Python: `pandas`, `scikit-learn`, `xgboost`
- Explainability: `SHAP`
- UI: `Streamlit`
- Visualization: `Plotly`, `Seaborn`, `Matplotlib`
- Data: `Kaggle` API

---
