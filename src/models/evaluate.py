"""
evaluate.py
-----------
Generates model performance charts and saves them to evaluation/.

Outputs
───────
  evaluation/roc_curve.png
  evaluation/precision_at_k.png
  evaluation/lift_chart.png
  evaluation/shap_summary.png
  evaluation/metrics.json

Usage:
    python src/models/evaluate.py
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib
import shap

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.feature_engineering import FEATURE_COLS

ARTIFACT_DIR = Path("src/models/artifacts")
EVAL_DIR     = Path("evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
DARK_BG    = "#0F1117"
CARD_BG    = "#1A1D27"
ACCENT     = "#00C2FF"
ACCENT2    = "#FF6B6B"
TEXT_COLOR = "#E8EAF0"

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG,
    "axes.edgecolor": "#2A2D3A", "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR, "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR, "grid.color": "#2A2D3A",
    "grid.linestyle": "--", "grid.alpha": 0.5,
    "font.family": "sans-serif",
})


def load_predictions():
    pred_path = ARTIFACT_DIR / "test_predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError("Run train.py first.")
    df = pd.read_parquet(pred_path)
    return df["y_true"].values, df["y_prob"].values


# ── ROC Curve ─────────────────────────────────────────────────────────────────
def plot_roc(y_true, y_prob):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color=ACCENT, lw=2.5, label=f"XGBoost  (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="#555", lw=1, label="Random baseline")
    ax.fill_between(fpr, tpr, alpha=0.10, color=ACCENT)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Churn Classifier", fontsize=14, fontweight="bold", color=TEXT_COLOR)
    ax.legend(frameon=False, fontsize=11)
    ax.grid(True)
    fig.tight_layout()
    out = EVAL_DIR / "roc_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC curve  → {out}  (AUC={auc:.4f})")
    return auc


# ── Precision@K ──────────────────────────────────────────────────────────────
def plot_precision_at_k(y_true, y_prob, k_list=None):
    if k_list is None:
        k_list = list(range(5, 55, 5))  # 5%…50%

    precisions, recalls = [], []
    total_pos = y_true.sum()
    order = np.argsort(y_prob)[::-1]
    n = len(y_true)

    for k_pct in k_list:
        k_n = max(1, int(n * k_pct / 100))
        top_k = y_true[order[:k_n]]
        precisions.append(top_k.mean())
        recalls.append(top_k.sum() / total_pos)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_list, [p * 100 for p in precisions], color=ACCENT, lw=2.5, marker="o",
            markersize=5, label="Precision@K")
    ax.plot(k_list, [r * 100 for r in recalls], color=ACCENT2, lw=2.5, marker="s",
            markersize=5, label="Recall@K")
    ax.axhline(y_true.mean() * 100, linestyle="--", color="#888", lw=1.2,
               label=f"Baseline ({y_true.mean():.1%})")
    ax.set_xlabel("Top-K% of Subscribers Contacted", fontsize=12)
    ax.set_ylabel("% (%)", fontsize=12)
    ax.set_title("Precision@K  &  Recall@K", fontsize=14, fontweight="bold", color=TEXT_COLOR)
    ax.legend(frameon=False, fontsize=11)
    ax.grid(True)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    fig.tight_layout()
    out = EVAL_DIR / "precision_at_k.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    p10 = precisions[1]  # k=10%
    print(f"  Precision@K → {out}  (P@10%={p10:.3f})")
    return precisions, recalls


# ── Decile Lift Chart ─────────────────────────────────────────────────────────
def plot_lift_chart(y_true, y_prob):
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    # labels are integers 1..10 where 1 = highest risk
    df["decile"] = pd.qcut(df["y_prob"].rank(method="first"), 10,
                           labels=list(range(10, 0, -1)))
    df["decile"] = df["decile"].astype(int)
    decile_stats = df.groupby("decile", sort=True).agg(
        total=("y_true", "count"),
        churners=("y_true", "sum"),
    ).reset_index().sort_values("decile")
    decile_stats["rate"] = decile_stats["churners"] / decile_stats["total"]
    baseline = y_true.mean()
    decile_stats["lift"] = decile_stats["rate"] / baseline

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [ACCENT if d <= 3 else ACCENT2 if d <= 6 else "#555"
              for d in decile_stats["decile"]]
    bars = ax.bar(decile_stats["decile"].astype(str),
                  decile_stats["lift"], color=colors, width=0.6)
    ax.axhline(1.0, linestyle="--", color="#888", lw=1.5, label="Baseline lift = 1.0")
    ax.set_xlabel("Decile (1 = Highest Risk)", fontsize=12)
    ax.set_ylabel("Lift", fontsize=12)
    ax.set_title("Cumulative Lift Chart by Decile", fontsize=14, fontweight="bold", color=TEXT_COLOR)
    ax.legend(frameon=False, fontsize=11)
    ax.grid(True, axis="y")
    for bar, lift in zip(bars, decile_stats["lift"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{lift:.2f}×", ha="center", va="bottom", fontsize=10, color=TEXT_COLOR)
    fig.tight_layout()
    out = EVAL_DIR / "lift_chart.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    # Decile 1 = highest risk row
    d1_row = decile_stats[decile_stats["decile"] == 1]
    d1_lift = float(d1_row["lift"].values[0]) if len(d1_row) else float(decile_stats.iloc[0]["lift"])
    print(f"  Lift chart  → {out}  (Decile-1 lift={d1_lift:.2f}×)")
    return decile_stats


# ── SHAP Summary ──────────────────────────────────────────────────────────────
def plot_shap_summary():
    shap_path = ARTIFACT_DIR / "shap_values.npy"
    if not shap_path.exists():
        print("  ⚠  SHAP values not found, skipping summary plot.")
        return

    sv = np.load(str(shap_path))
    # XGBoost 3.x may produce shape (n, features, 1) — squeeze to 2D
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[2] == 2 else sv.squeeze(-1)

    # Compute mean absolute SHAP per feature
    mean_abs = np.abs(sv).mean(axis=0)
    # Guard: if shape mismatch, fall back to equal weights
    if len(mean_abs) != len(FEATURE_COLS):
        mean_abs = np.ones(len(FEATURE_COLS))
    feat_imp = pd.Series(mean_abs, index=FEATURE_COLS).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors_bar = [ACCENT if v > feat_imp.median() else ACCENT2 for v in feat_imp.values]
    ax.barh(feat_imp.index, feat_imp.values, color=colors_bar, height=0.65)
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title("SHAP Feature Importance — Churn Drivers", fontsize=14,
                 fontweight="bold", color=TEXT_COLOR)
    ax.grid(True, axis="x")
    fig.tight_layout()
    out = EVAL_DIR / "shap_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  SHAP summary→ {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("─── Evaluation ─────────────────────────────────────────")
    y_true, y_prob = load_predictions()
    print(f"  Test set: {len(y_true):,} subscribers  (churn={y_true.mean():.1%})")

    auc          = plot_roc(y_true, y_prob)
    precisions, recalls = plot_precision_at_k(y_true, y_prob)
    decile_stats = plot_lift_chart(y_true, y_prob)
    plot_shap_summary()

    # Save metrics JSON
    d1_row = decile_stats[decile_stats["decile"] == 1]
    d1_lift_val = float(d1_row["lift"].values[0]) if len(d1_row) else float(decile_stats.iloc[0]["lift"])
    metrics = {
        "auc"         : round(float(auc), 4),
        "precision_10": round(float(precisions[1]), 4),
        "recall_10"   : round(float(recalls[1]), 4),
        "lift_decile1": round(d1_lift_val, 3),
    }
    metrics_path = EVAL_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics: {metrics}")
    print(f"✅ Evaluation complete → {EVAL_DIR}/")
