import os
import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42

# --- Paths (relative to notebooks/*/*) ---
TRAIN_PATH = "../../data/train.csv"
TEST_PATH  = "../../data/test.csv"
SAMPLE_PATH= "../../data/sample_submission.csv"

OUT_BASE = "../../outputs"
OUT_HOLD = f"{OUT_BASE}/holdout_reports"
OUT_IMP  = f"{OUT_BASE}/feature_importance"
OUT_SUB  = f"{OUT_BASE}/submissions"

# --- IO ---
def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH, sample_path=SAMPLE_PATH):
    """Load train, test, and sample_submission CSV files."""
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    return train, test, sample

def infer_target(train: pd.DataFrame, test: pd.DataFrame) -> str:
    """Infer target column name (present in train but not in test)."""
    for col in train.columns:
        if col not in test.columns:
            return col
    return train.columns[-1]

def split_features(train: pd.DataFrame, target_col: str):
    """Split features into numeric and categorical lists."""
    feature_cols = [c for c in train.columns if c != target_col]
    cat_cols = [c for c in feature_cols if train[c].dtype == 'object']
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return feature_cols, cat_cols, num_cols

# --- Preprocessing ---
def make_preprocessor(num_cols, cat_cols, scale_numeric: bool):
    """Build preprocessing pipeline with imputation + scaling/OHE."""
    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    num_steps = [("imp", numeric_imputer)]
    if scale_numeric:
        num_steps.append(("sc", StandardScaler()))

    preprocess = ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", Pipeline([("imp", categorical_imputer), ("oh", onehot)]), cat_cols)
    ])
    return preprocess

# --- Holdout Split ---
def holdout_split(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """Split the dataset into training and validation parts."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# --- Metrics ---
def eval_regression(y_true, y_pred):
    """Compute RMSE, MAE, and R² for regression models."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

# --- Explainability ---
def get_feature_names_from_pipe(pipe, num_cols, cat_cols):
    """Extract expanded feature names after OneHot encoding."""
    try:
        oh = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["oh"]
        cat_feature_names = []
        for col, cats in zip(cat_cols, oh.categories_):
            cat_feature_names.extend([f"{col}={c}" for c in cats])
        feature_names = list(num_cols) + cat_feature_names
        return feature_names
    except Exception:
        return list(num_cols) + list(cat_cols)

def permutation_importance_df(pipe, X_valid, y_valid, num_cols, cat_cols, n_repeats=5):
    """Compute permutation feature importance (by RMSE drop)."""
    perm = permutation_importance(
        pipe, X_valid, y_valid,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error"
    )
    names = get_feature_names_from_pipe(pipe, num_cols, cat_cols)
    n = min(len(names), len(perm.importances_mean))
    df = pd.DataFrame({
        "feature": names[:n],
        "perm_importance_rmse": perm.importances_mean[:n]
    }).sort_values("perm_importance_rmse", ascending=False).reset_index(drop=True)
    return df

# --- Saving helpers ---
def save_json(obj, path):
    """Save a dictionary to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_csv_df(df, path):
    """Save a DataFrame to CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def save_submission(sample_df, preds, out_name="submission.csv"):
    """Save model predictions as a Kaggle submission CSV."""
    pred_cols = [c for c in sample_df.columns if c != sample_df.columns[0]]
    pred_col = pred_cols[0] if pred_cols else "target"
    sub = sample_df.copy()
    sub[pred_col] = np.clip(preds, 0, 1)
    out_path = f"{OUT_SUB}/{out_name}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sub.to_csv(out_path, index=False)
    return out_path
