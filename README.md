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
Executive Summary: Ames Housing Market Predictive Analytics & Machine Learning Pipeline
Ames Housing Market Analysis & Predictive Modeling: Executive Summary
1. Executive Overview & Strategic Problem Statement

In the high-stakes environment of residential real estate, the ability to generate high-fidelity property valuations is a cornerstone of financial risk mitigation and competitive positioning. For institutional stakeholders in the Ames, Iowa market, accurate predictive modeling serves as a critical hedge against over-leveraged acquisitions and capital misallocation. This project establishes a robust analytical framework by engineering a high-precision machine learning pipeline designed to synthesize a complex 81-feature dataset into actionable, low-variance price forecasts.

The primary outcome of the project was the development of a sophisticated gradient boosting architecture that achieved a peak experimental R^{2} performance of 0.9122.

This predictive capability transforms raw market data into a scalable strategic asset, providing a rigorous foundation for automated valuation. To achieve this level of accuracy, the investigation leveraged a comprehensive dataset encompassing granular physical structural attributes alongside critical neighborhood-level metadata.

2. Data Ingestion & Integrity Audit

The reliability of any model inference is strictly bounded by the health of the underlying data. A comprehensive audit was conducted to identify missing signals and distributional anomalies that could compromise algorithmic performance if left uncorrected.

Dataset Snapshot

Metric	Statistic
Total Observations	1,460 Rows
Initial Feature Count	81 Columns
Total Null Values	7,829
Target Variable (SalePrice) Mean	$180,921
Target Variable (SalePrice) Median	$163,000
Target Variable (SalePrice) Skew	1.88

The audit revealed a significant target variable skew of 1.88, indicating a right-tailed distribution where a small subset of luxury properties disproportionately influences the mean. To ensure the model remains sensitive to the full spectrum of market values and to minimize error variance in high-end valuations, a Log-Transformation was identified as the essential mathematical remediation to normalize the target distribution. This diagnostic phase transitioned the project from raw data ingestion to domain-aware remediation.

3. Domain-Aware Cleaning & Feature Engineering

Rather than utilizing generic automated imputation, this project applied "Domain-Aware" logic to preserve the structural signals within the data. This approach distinguishes between truly missing information and the intentional absence of a feature (e.g., a null value for a pool indicating no pool exists).

Categorized Engineering Interventions

* Missing Value Remediation:
  * Categorical Logic: Features such as PoolQC, MiscFeature, FireplaceQu, and MasVnrType were assigned a value of "None" to accurately reflect the absence of these amenities.
  * Numerical Logic: Nulls in area-specific features including GarageArea, BsmtFinSF, and MasVnrArea were remediated to "0," indicating a zero-square-foot footprint for nonexistent structures.
* Derived Predictive Signals:
  * Temporal & Quality Dynamics: Beyond the raw Age feature (correlation: -0.52), we engineered IsRemodeled to capture value-add interventions and RemodAge to track depreciation post-renovation.
  * Utility & Capacity: Composite features were developed to capture total property utility, specifically TotalSF (Basement + 1st/2nd floor), TotalBath (weighted sum), TotalPorchSF, and a binary HasPool indicator.

The strategic value of these engineered features is underscored by the property Age variable, which provides a direct, linear signal of depreciation (-0.52 correlation with SalePrice) far superior to raw timestamps. These refined inputs provided the high-signal foundation required for the machine learning architecture.

4. Machine Learning Pipeline & Technical Architecture

To ensure scalability and prevent data leakage, the project implemented a dynamic ColumnTransformer pipeline. This architecture standardizes the preprocessing workflow, allowing for the seamless ingestion of new market data.

ColumnTransformer Architecture

1. Numerical Path: Utilizes StandardScaler to normalize features, ensuring that large-scale variables (e.g., TotalSF) do not exert disproportionate influence over gradient descent compared to smaller ordinal metrics.
2. Categorical Path: Employs OneHotEncoder with handle_unknown='ignore'.

While the engineered dataset contains 88 base features, the categorical encoding process expands the feature space into a significantly higher-dimensional matrix of "dummy" variables. This expansion highlights the necessity of utilizing Gradient Boosting Decision Trees, which excel at maintaining computational efficiency and preventing overfitting in high-dimensional tabular environments. The model was evaluated using a strict 80/20 training/test split strategy to ensure the integrity of performance metrics.

5. Model Development & Benchmark Performance

Gradient Boosting Decision Trees (GBDT) were selected as the primary candidates due to their ability to model complex, non-linear relationships in residential housing data. We benchmarked the two industry-standard architectures: XGBoost and LightGBM.

Model Benchmark Comparison

Model Name	MAE	RMSE	R^{2}
XGBoost Baseline	$15,629	$26,301	0.9098
LightGBM Baseline	$16,679	$28,553	0.8937

XGBoost was selected as the lead architecture for further refinement, as it demonstrated superior baseline precision by capturing nearly 91% of market variance with the lowest absolute error. While these baseline results were strong, advanced optimization was required to address residual errors in extreme price brackets.

6. Advanced Optimization & Model Refinement

Strategic refinements were focused on squeezing the final margin of performance from the XGBoost architecture through target transformation and exhaustive hyperparameter tuning.

Mathematical Transformation	Hyperparameter Tuning (GridSearchCV)
Log-Targeting: Applying a Log-Transformation to SalePrice reduced the impact of skew, successfully boosting peak experimental performance to 0.9122 R^{2}.	Optimal Parameters:<br>• Learning Rate: 0.1<br>• Max Depth: 4<br>• N-Estimators: 500<br>• Subsample: 0.8

The robustness of the tuned architecture was confirmed via 5-fold cross-validation, yielding a Mean R^{2} of 0.8905. It is critical to distinguish between the Peak Experimental R^{2} (0.9122) and the Final Validated R^{2} of 0.9082 achieved on the holdout test set. This validation ensures the model remains a reliable general-purpose tool rather than one overfitted to training noise.

7. Model Diagnostics & Market Insights

The "Explainability" layer of this project utilizes SHAP (SHapley Additive exPlanations) to identify the global impactors of property value, ensuring that model outputs are grounded in real-world market logic.

Top Market Drivers (SHAP Confirmed)

* Quality & Space: SHAP analysis identifies OverallQual (0.791 correlation) and GrLivArea (0.709 correlation) as the dominant global impactors of price.
* Physical Depreciation: The property Age (-0.523 correlation) remains the primary negative driver, highlighting the velocity of value decay in the absence of remodeling.

Model Diagnostics & Risk Management

* Mean Residual: $1,448 (indicating minimal global bias).
* Final Tuned R^{2} (Holdout): 0.9082.
* Residual Analysis:
  * Errors are tightly clustered around zero for mid-market properties.
  * Max Under-prediction: $166,946.
  * Max Over-prediction: -$129,071.
  * Variance increases at the $1.4M+ price extremes, necessitating a higher "confidence interval" for ultra-luxury assets.

The model demonstrates exceptional reliability for the vast majority of the Ames housing stock. With a final holdout R^{2} exceeding 0.90, this pipeline is ready for deployment as an automated valuation tool via the exported artifact ames_xgb_model.pkl.

