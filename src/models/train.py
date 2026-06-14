"""
train.py
--------
Trains an XGBoost churn classifier with SHAP explainability.

Steps
─────
1. Load processed feature Parquet
2. Train/test split (stratified 80/20)
3. XGBClassifier with scale_pos_weight for class imbalance
4. 5-fold stratified cross-validation
5. SHAP TreeExplainer → per-subscriber top-3 reason codes
6. Save model + scaler artifacts

Target metrics
──────────────
    AUC  > 0.88
    Precision@10%  > 0.60

Usage:
    python src/models/train.py
"""

import sys
import json
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.feature_engineering import FEATURE_COLS, TARGET_COL
from src.utils import shap_patch

PROCESSED_DIR = Path("data/processed")
ARTIFACT_DIR  = Path("src/models/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    feat_path = PROCESSED_DIR / "features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Run etl_pipeline.py first: {feat_path}")
    df = pd.read_parquet(feat_path)
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    return X, y, df


# ── Train ──────────────────────────────────────────────────────────────────────
def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Handle class imbalance
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    print(f"  Class balance — neg:{neg:,}  pos:{pos:,}  scale_pos_weight:{scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators          = 400,
        max_depth             = 6,
        learning_rate         = 0.05,
        subsample             = 0.80,
        colsample_bytree      = 0.80,
        scale_pos_weight      = scale_pos_weight,
        eval_metric           = "auc",
        early_stopping_rounds = 30,
        random_state          = 42,
        n_jobs                = -1,
        verbosity             = 0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    return model, X_train, X_test, y_train, y_test


# ── Cross-validate ────────────────────────────────────────────────────────────
def cross_validate(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.80, colsample_bytree=0.80,
        scale_pos_weight=np.bincount(y)[0] / np.bincount(y)[1],
        eval_metric="auc",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    scores = cross_val_score(cv_model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  5-fold CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


# ── SHAP explanations ─────────────────────────────────────────────────────
def compute_shap(model, X_test):
    print("  Computing SHAP values …")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test)
    # XGBoost 3.x TreeExplainer may return shape (n, features, 2) for binary
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv = sv[:, :, 1]   # take positive-class SHAP values
    elif isinstance(sv, list):
        sv = sv[1]         # old SHAP API: list[class_idx]
    return explainer, sv


def top_reason_codes(shap_row: np.ndarray, feature_names: list, top_k: int = 3) -> list:
    """Return top-k feature names sorted by absolute SHAP impact (descending)."""
    indices = np.argsort(np.abs(shap_row))[::-1][:top_k]
    return [feature_names[i] for i in indices]


# ── Save artifacts ────────────────────────────────────────────────────────────
def save_artifacts(model, feature_names):
    model_path = ARTIFACT_DIR / "churn_model.pkl"
    feat_path  = ARTIFACT_DIR / "feature_names.json"

    joblib.dump(model, model_path)
    with open(feat_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"  Model   → {model_path}")
    print(f"  Features→ {feat_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("─── Model Training ─────────────────────────────────────")

    X, y, df = load_data()
    print(f"  Dataset: {X.shape[0]:,} rows × {X.shape[1]} features")

    model, X_train, X_test, y_train, y_test = train(X, y)

    # Evaluation
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"  Test AUC: {auc:.4f}", "✅" if auc > 0.88 else "⚠ (below 0.88 target)")

    # Cross-validation
    cross_validate(model, X, y)

    # SHAP
    _, shap_values = compute_shap(model, X_test)

    # Persist
    save_artifacts(model, FEATURE_COLS)

    # Also save test predictions for evaluate.py
    pred_df = pd.DataFrame({
        "y_true"    : y_test,
        "y_prob"    : y_prob,
    })
    pred_df.to_parquet(ARTIFACT_DIR / "test_predictions.parquet", index=False)

    np.save(str(ARTIFACT_DIR / "shap_values.npy"), shap_values)
    print(f"\n✅ Training complete.  AUC = {auc:.4f}")
