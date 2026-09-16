# 🏡 Ames Housing — Exploratory Data Analysis

A complete Exploratory Data Analysis (EDA) of the **Ames Housing dataset** (2,930 home sales in Ames, Iowa, 2006–2010), executed in Python using pandas, matplotlib, and seaborn.

---

## 📌 Project Overview

The Ames Housing dataset is a rich, real-world dataset with 80 features describing residential home sales. This project performs a full EDA to understand:

- The distribution of the target variable (`SalePrice`)
- How individual features behave
- Which features correlate most strongly with sale price
- How engineered features (like home `Age`) reveal hidden patterns

The goal is to build a strong data-understanding foundation before any machine learning model is applied.

---

## 🗂️ Repository Contents

| File | Description |
|------|-------------|
| `ameshousing.ipynb` | The main Colab notebook containing the full analysis |
| `README.md` | This file — project overview and findings |

---

## 🛠️ Tools & Libraries

- **Python 3.x**
- **pandas** — data loading, cleaning, aggregation
- **numpy** — numerical operations
- **matplotlib** — static visualizations
- **seaborn** — statistical visualizations (box plots)
- **Google Colab** — development environment

---

## 🔍 What Was Done

The EDA is broken into five main sections:

### 1. Data Loading & Cleaning
- Loaded the raw `AmesHousing.csv`
- Dropped redundant `Order` column
- Set `PID` as the index
- Standardized column names (removed spaces for consistency)

### 2. Distribution Analysis
Examined three key columns:
- **SalePrice** — right-skewed, mean > median, high variance
- **TotRmsAbvGrd** (total rooms above grade) — roughly symmetric, centered at 6
- **OverallCond** (overall condition) — concentrated around "average" rating of 5

### 3. Subset Analysis
Split data by `OverallCond` into below-average, average, and above-average condition groups — then compared their `SalePrice` distributions.

### 4. Correlation Analysis
Identified the most positively and most negatively correlated features with `SalePrice` using Pearson correlation.

### 5. Feature Engineering
Created a new `Age` feature (`YrSold − YearBuilt`) and visualized its relationship with sale price.

---

## 📊 Key Findings

### 🔹 SalePrice Distribution
- **Mean:** ~$180,900
- **Median:** ~$160,000
- **Standard Deviation:** ~$80,000
- **Skewness:** Positive (~1.7) → right-skewed
- **Insight:** Most homes sell between $100K–$250K; a small number of luxury homes (up to $755K) inflate the mean.

### 🔹 Feature Correlations
| Feature | Correlation with SalePrice | Interpretation |
|---------|---------------------------|----------------|
| `OverallQual` | **+0.80** | Strongest predictor — quality matters most |
| `GrLivArea` | **+0.70** | Larger living areas → higher price |
| `GarageCars` | **+0.65** | Bigger garages correlate with pricier homes |
| `MoSold` | **−0.05** | Almost no relationship — month doesn't matter |
| `YrSold` | **−0.03** | Slight negative — market timing weak |

### 🔹 Surprising Insight — Condition ≠ Price
Splitting by `OverallCond` produced **heavily overlapping** SalePrice distributions. Condition is a **weak standalone predictor** — quality (`OverallQual`) is far more reliable.

### 🔹 Home Age vs Price
- Correlation: **−0.50** (moderate negative)
- Older homes tend to sell for less, **but with huge variance** — historic homes in desirable neighborhoods still command high prices.

### 🔹 The Value of Log-Transformation
Applying `np.log(SalePrice)` reduced skewness from ~1.7 to ~0.0, producing a near-perfect normal distribution. This is a **critical preprocessing step** for any linear model.

---

## 📈 Visualizations Included

- Histograms for `SalePrice`, `TotRmsAbvGrd`, `OverallCond`
- Overlapping histograms of `SalePrice` grouped by condition
- Box plots of `SalePrice` vs most correlated features
- Scatter plot of home `Age` vs `SalePrice`
- Side-by-side comparison of original vs log-transformed `SalePrice`

---

## 🎓 What I Learned

1. **EDA is not optional** — 80% of real ML work happens before modeling
2. **Correlation ≠ Causation** — but it's a great first filter for feature selection
3. **Feature engineering matters** — raw columns like `YrSold` and `YearBuilt` are useless alone, but combining them into `Age` reveals a pattern
4. **Skewness kills models** — log-transforming the target is a must for regression
5. **Not all "numeric" columns are numbers** — `OverallCond` is encoded 1–10 but behaves categorically

---

## 🚀 Future Work

- Clean null values using domain-aware imputation
- Apply log-transformation to SalePrice for modeling
- Build regression models (Linear, Ridge, XGBoost)
- Perform feature selection using mutual information and SHAP
- Deploy a simple price-prediction web app via Streamlit

---

## 🔗 How to Run This Notebook

1. Open the notebook in **Google Colab** (badge below)
2. Run the first cell to download the dataset
3. Execute cells top-to-bottom

Or clone this repo and run locally:

```bash
git clone https://github.com/ShachiPradhan/ames-housing.git
cd ames-housing
jupyter notebook ameshousing.ipynb
---




