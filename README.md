# 🛣️ Kaggle Predict Accident Risk (Playground Series S5E10)

**Goal:** Predict the *accident risk* — a continuous value between **0 and 1** — for different types of roads.  
This project is part of **Kaggle Playground Series — Season 5, Episode 10**, and focuses on a clean, interpretable **regression baseline**, 
ideal for a data science portfolio.

---

## 📂 Project Structure

```
ps_s5e10_regression_project/
├── data/ # place train.csv, test.csv, sample_submission.csv here
├── common/
│ └── prep.py # shared functions: loading, preprocessing, metrics, saving
├── notebooks/
│ ├── 1_eda/
│ │ └── 1_eda.ipynb # EDA for regression: preview, target histogram, missing values, feature split
│ ├── 2_ridge/
│ │ └── ridge.ipynb # Ridge regression baseline
│ ├── 3_random_forest/
│ │ └── random_forest_reg.ipynb # RandomForestRegressor experiment
│ ├── 4_hist_gb/
│ └── hist_gb_reg.ipynb # HistGradientBoostingRegressor experiment 
├── outputs/
│ ├── holdout_reports/ # *_holdout.json (RMSE, MAE, R²)
│ ├── feature_importance/ # *_perm_importance.csv (Permutation Importance)
│ └── submissions/ # *_reg.csv (ready Kaggle submissions)
└── README.md

```

---

## 📊 Project Overview

This repository provides a reproducible workflow for predicting accident risk using regression models.  
Each notebook is self-contained, with concise code, clear markdown explanations, and auto-saving of results.

---

## 🧠 Models Used

- **Ridge Regression** → linear baseline with scaled numeric features  
- **RandomForestRegressor** → strong tree-based non-linear model  
- **HistGradientBoostingRegressor** → efficient gradient boosting approach for tabular data  

**Evaluation metrics:**
- RMSE — Root Mean Squared Error  
- MAE — Mean Absolute Error  
- R² — Coefficient of Determination  

Each model is evaluated using a simple **holdout validation (80/20 split)**.

---

## 🧩 Workflow

1. **EDA** — explore the dataset, inspect distributions, detect missing values  
2. **Model Training** — run each model notebook; metrics are automatically saved under `/outputs/holdout_reports`  
3. **Feature Importance** — computed via permutation importance on the holdout set; they are automatically saved under `/outputs/feature_importance`  
4. **Submission** — best model predictions are saved in `/outputs/submissions/` ready for Kaggle upload  

---

## 🏁 Next Steps

- Hyperparameter tuning for RandomForest and HGB models  
- Advanced feature engineering (ratios, polynomial terms, interaction effects)  
- Model ensembling or stacking for further accuracy improvements  

---

## 📎 Kaggle Competition

🔗 [Playground Series — Season 5, Episode 10](https://www.kaggle.com/competitions/playground-series-s5e10)

---

## 📈 Results Summary

| Model | RMSE | MAE | R² | Notes |
|--------|------|-----|----|-------|
| **Ridge Regression** | 0.0735 | 0.0583 | 0.8042 | Simple linear baseline; light and interpretable, but lacks non-linear power |
| **Random Forest Regressor** | 0.0575 | 0.0446 | 0.8801 | Strong non-linear model with good accuracy, but slower and more resource-heavy |
| **HistGradientBoosting Regressor** | **0.0564** | **0.0437** | **0.8850** | Best overall — high accuracy and efficient training thanks to histogram optimization |

---

### 🏁 Model Ranking

| Rank | Model | 
|------|--------|
| 🥇 **HistGradientBoosting Regressor** | 
| 🥈 **Random Forest Regressor** | 
| 🥉 **Ridge Regression** | 

---

### 💡 Conclusions

- **HistGradientBoosting** outperforms all other models in both **efficiency** and **predictive power**.  
- **Random Forest** remains a reliable, interpretable benchmark but is slower to train.  
- **Ridge Regression** provides a simple linear baseline that’s useful for comparison,  
  though it fails to capture complex relationships in the data.  

✅ Final model choice: **HistGradientBoosting Regressor**

### 💡 Notes

This baseline is designed to be:
- **Readable:** clear notebook structure and markdown commentary  
- **Modular:** shared preprocessing and metrics in `common/prep.py`  
- **Reproducible:** automatic saving of metrics, importances, and submissions in `/outputs/`
