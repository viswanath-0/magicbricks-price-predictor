# Magicbricks Property Analytics & Price Predictor

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=flat&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

An end-to-end data science portfolio project covering web scraping, exploratory data analysis, advanced statistics, and a full machine learning pipeline that predicts Indian real estate prices using a Bayesian-tuned Gradient Boosting Regressor.

**Live Demo:** [magicbricks-price-predictor.streamlit.app](https://magicbricks-price-predictor-viswanath.streamlit.app)

---

## Project Overview

| Phase | What was done |
|-------|--------------|
| Web Scraping | Collected 5,650 property listings across 6 Indian cities using `requests` and `BeautifulSoup` with regex extraction |
| Data Cleaning | Handled missing values, dropped unusable columns, filtered outliers, engineered `price_per_sqft` and `price_cr` |
| EDA | Univariate and bivariate analysis with 8 interactive Plotly charts |
| Advanced Statistics | Skewness, kurtosis, five-number summary, outlier detection (IQR / Z-score / Modified Z-score), normality testing, hypothesis tests |
| ML Pipeline | Feature selection (4 methods), 7 models trained, 3 tuning strategies, cross-validation, bias-variance analysis |
| Web Application | Professional dark-themed Streamlit app with 3D card effects, live price predictor, and model comparison dashboard |

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | Magicbricks.com (scraped) |
| Raw size | 5,650 listings |
| Clean size | 5,019 listings |
| Cities | Hyderabad, Pune, Chennai, Mumbai, Gurgaon, Noida |
| Features used | BHK, Area (sqft), City, Furnishing, Status |
| Target | Price (INR) — log-transformed for training |

---

## Machine Learning Pipeline

### 1. Preprocessing

- Median imputation for BHK
- Mode imputation for categorical features (Furnishing, Status)
- `OrdinalEncoder` for City, Furnishing, Status
- `log1p` transform on target (price is right-skewed)

### 2. Feature Selection — 4 methods, consensus voting

| Method | Type |
|--------|------|
| SelectKBest with f_regression | Filter — ANOVA F-statistic |
| Mutual Information Regression | Filter — information-theoretic |
| Recursive Feature Elimination (LinearRegression) | Wrapper |
| GBM Feature Importances | Embedded |

Features selected by 3 or more methods are kept for training.

### 3. Models Trained & Compared

| Model | Type | Role |
|-------|------|------|
| Decision Tree | Non-parametric | Baseline |
| Ridge Regression | Parametric | Regularised linear baseline |
| Random Forest | Non-parametric Ensemble | 100 independent trees, averaged |
| GBM Baseline | Non-parametric Ensemble | Default hyperparameters |
| GBM GridSearchCV | Tuned Ensemble | Best of 24 grid combinations |
| GBM RandomizedSearchCV | Tuned Ensemble | Best of 40 random samples |
| **GBM Bayesian (Optuna)** | **Tuned Ensemble** | **Best of 40 Bayesian trials — final model** |

### 4. Hyperparameter Tuning

- **GridSearchCV** — exhaustive search over a fixed parameter grid
- **RandomizedSearchCV** — random sampling across a wider distribution
- **Optuna (Bayesian)** — intelligent search using past trial outcomes to find optimal parameters

### 5. Cross Validation

- KFold (k = 5)
- KFold (k = 10)
- RepeatedKFold (5 splits × 3 repeats = 15 evaluations)

### 6. Bias-Variance Analysis

- sklearn `learning_curve` with R² scoring
- mlxtend `bias_variance_decomp` with MSE loss
- mlxtend `plot_learning_curves`

### Why Gradient Boosting was chosen

- Handles mixed categorical and numerical features — no scaling required
- Captures non-linear interactions (e.g. BHK × City price multipliers)
- Sequential error correction — each tree specifically fixes residuals from the previous one
- Bayesian tuning (Optuna) finds better hyperparameters than exhaustive grid search

---

## Key Results

| Metric | Value |
|--------|-------|
| Best model | GBM Bayesian (Optuna) |
| R² — test set | ~0.72 – 0.76 |
| Cross-validation mean R² | ~0.70 – 0.74 |
| MAE | ~0.50 – 0.70 Cr |
| Top feature | area_sqft |

---

## Web Application

Dark-themed Streamlit app with 3D card effects and interactive Plotly charts.

**4 tabs:**

| Tab | Contents |
|-----|----------|
| Overview | Dataset KPIs, city summary table, furnishing pie chart, sample data |
| EDA & Statistics | 8 interactive charts, skewness table, outlier counts, hypothesis test cards |
| Model Performance | Pipeline visual, all 7 models ranked by R², actual vs predicted scatter, CV chart, feature importance, why-GBM cards |
| Price Predictor | Input form, predicted price card, city percentile gauge, feature contribution chart, similar listings |

---

## Project Structure

```
magicbricks-price-predictor/
├── app.py                               # Streamlit web application (1,140 lines)
├── magicbricks_properties.xls           # Scraped dataset (CSV format, 5,019 rows)
├── Magicbricks_Complete_Pipeline.ipynb  # Full notebook — scraper + EDA + stats + ML
├── requirements.txt                     # App dependencies
├── README.md
└── .streamlit/
    └── config.toml                      # Dark theme configuration
```

---

## Run Locally

```bash
# 1. Clone
git clone https://github.com/viswanath-0/magicbricks-price-predictor.git
cd magicbricks-price-predictor

# 2. Virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

Opens at `http://localhost:8501`

> First load takes ~30 seconds for model training. After that `@st.cache_resource` keeps everything cached — tab switches are instant.

---

## Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
scikit-learn>=1.3.0
scipy>=1.11.0
```

---

## Skills Demonstrated

`Python` `Web Scraping` `BeautifulSoup` `Regex` `Pandas` `NumPy` `Scikit-learn` `Plotly` `Streamlit` `Feature Engineering` `Feature Selection` `Ensemble Methods` `Gradient Boosting` `Hyperparameter Tuning` `GridSearchCV` `RandomizedSearchCV` `Optuna` `Cross Validation` `Bias-Variance Analysis` `Hypothesis Testing` `Statistical Analysis` `Data Visualisation` `Git` `Streamlit Cloud`

---

## Author

**Viswanath** — Data Science Portfolio

[![GitHub](https://img.shields.io/badge/GitHub-viswanath--0-181717?style=flat&logo=github)](https://github.com/viswanath-0)
