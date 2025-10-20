# 🛣️ Kaggle Predict Accident Risk (Playground Series S5E10)

**Goal:** Predict the likelihood of accidents on different types of roads.  
This project is part of the **Kaggle Playground Series — Season 5, Episode 10**.  
The focus is on building a **clean, interpretable baseline** using simple, well-documented models — ideal for a junior-level portfolio.

---

## 📊 Project Overview

This repository contains a reproducible workflow for exploring and modeling accident risk prediction:

| Folder | Description |
|--------|--------------|
| `data/` | Training, test, and sample submission files |
| `notebooks/1_eda/` | Exploratory Data Analysis — data understanding and target inspection |
| `notebooks/2_logreg/`, `3_random_forest/`, `4_hist_gb/` | Individual model experiments |
| `notebooks/5_compare/` | Model comparison and leaderboard summary |
| `outputs/` | Automatically saved metrics, feature importances, and submissions |

---

## 🧠 Models Used
- **Logistic Regression** → simple, interpretable baseline  
- **Random Forest Classifier** → strong tree-based benchmark  
- **HistGradientBoostingClassifier** → efficient gradient boosting model  

**Evaluation metric:** ROC AUC (Stratified 5-Fold Cross-Validation)

---

## 🧩 Project Workflow
1. **EDA** — inspect data, check balance, identify missing values  
2. **Model Training** — run each model notebook; results auto-save to `/outputs`  
3. **Comparison** — open `5_compare/compare_models.ipynb` for a performance summary  
4. **Submission** — upload the best CSV from `/outputs/submissions` to Kaggle  

---

## 🏁 Next Steps
- Hyperparameter tuning for top-performing models  
- Feature engineering (ratios, binning, interactions)  
- Calibration of predicted probabilities  

---

## 📎 Kaggle Competition
🔗 [Playground Series — Season 5, Episode 10](https://www.kaggle.com/competitions/playground-series-s5e10)
