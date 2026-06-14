"""
streamlit_app.py
----------------
Streamlit dashboard for the Telecom Churn Prediction Engine.
Shows top-100 at-risk subscribers, KPI summary cards, and interactive
SHAP feature-importance waterfall per subscriber.

Run:
    streamlit run src/dashboard/streamlit_app.py
"""

import sys
import json
import math
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

# ── Project imports ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from src.features.feature_engineering import engineer_features, FEATURE_COLS
from src.api.schemas import build_response, REASON_LABELS, RETENTION_ACTIONS
from src.utils import shap_patch

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Radar · Banglalink CVM",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0F1117; }

.kpi-card {
    background: linear-gradient(135deg, #1A1D27 0%, #12151F 100%);
    border: 1px solid #2A2D3A;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: transform 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-value { font-size: 2.2rem; font-weight: 700; color: #00C2FF; line-height: 1.2; }
.kpi-label { font-size: 0.82rem; color: #8A8FAA; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-delta { font-size: 0.82rem; color: #FF6B6B; margin-top: 2px; }

.risk-HIGH   { color: #FF6B6B; font-weight: 700; }
.risk-MEDIUM { color: #FFB347; font-weight: 600; }
.risk-LOW    { color: #4CAF50; }

div[data-testid="stSidebar"] {
    background: #12151F;
    border-right: 1px solid #2A2D3A;
}

.section-header {
    font-size: 1.1rem; font-weight: 600; color: #E8EAF0;
    padding: 8px 0 4px; border-bottom: 1px solid #2A2D3A; margin-bottom: 12px;
}

.stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load model + data ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading churn model …")
def load_model():
    model_path = BASE_DIR / "src/models/artifacts/churn_model.pkl"
    feat_path  = BASE_DIR / "src/models/artifacts/feature_names.json"
    if not model_path.exists():
        return None, None
    model = joblib.load(model_path)
    with open(feat_path) as f:
        feat_names = json.load(f)
    return model, feat_names


@st.cache_data(show_spinner="Scoring subscribers …", ttl=300)
def load_scored_data():
    feat_path = BASE_DIR / "data/processed/features.parquet"
    if not feat_path.exists():
        return None

    model, feat_names = load_model()
    if model is None:
        return None

    df = pd.read_parquet(feat_path)
    X  = df[feat_names].values
    probs = model.predict_proba(X)[:, 1]
    df["churn_prob"] = probs.round(4)

    def tier(p):
        if p >= 0.65: return "HIGH"
        if p >= 0.35: return "MEDIUM"
        return "LOW"
    df["risk_tier"] = df["churn_prob"].apply(tier)
    return df


@st.cache_data(show_spinner=False)
def load_metrics():
    m_path = BASE_DIR / "evaluation/metrics.json"
    if m_path.exists():
        with open(m_path) as f:
            return json.load(f)
    return {}


# ── SHAP waterfall ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing SHAP …")
def compute_shap_row(_model, X_row: np.ndarray, feat_names: list):
    try:
        import shap
        explainer = shap.TreeExplainer(_model)
        sv = explainer.shap_values(X_row)[0]
        return sv
    except Exception:
        return _model.feature_importances_


def shap_waterfall_chart(sv: np.ndarray, feat_names: list, base_value: float = 0.0):
    pairs = sorted(zip(feat_names, sv), key=lambda x: abs(x[1]), reverse=True)[:12]
    names = [p[0] for p in pairs][::-1]
    vals  = [p[1] for p in pairs][::-1]
    colors = ["#FF6B6B" if v > 0 else "#00C2FF" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside",
        textfont=dict(color="#E8EAF0", size=11),
    ))
    fig.update_layout(
        title=dict(text="SHAP Feature Impact (red = churn driver, blue = retention signal)",
                   font=dict(color="#E8EAF0", size=13)),
        paper_bgcolor="#0F1117", plot_bgcolor="#1A1D27",
        font=dict(color="#E8EAF0"),
        xaxis=dict(title="SHAP Value", gridcolor="#2A2D3A", zeroline=True,
                   zerolinecolor="#555"),
        yaxis=dict(gridcolor="#2A2D3A"),
        margin=dict(l=160, r=80, t=50, b=40),
        height=400,
    )
    return fig


# ── Main dashboard ─────────────────────────────────────────────────────────────
def main():
    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📡 Churn Radar")
        st.caption(f"Banglalink CVM · {datetime.now().strftime('%d %b %Y')}")
        st.divider()

        risk_filter = st.selectbox(
            "Risk Tier Filter",
            ["ALL", "HIGH", "MEDIUM", "LOW"],
            index=0,
        )
        top_n = st.slider("Show Top N Subscribers", 10, 500, 100, step=10)
        segment_filter = st.multiselect(
            "Segment", ["Prepaid", "Postpaid", "Hybrid"],
            default=["Prepaid", "Postpaid", "Hybrid"]
        )

        st.divider()
        metrics = load_metrics()
        if metrics:
            st.markdown("**Model Performance**")
            st.metric("ROC-AUC", f"{metrics.get('auc', 0):.4f}")
            st.metric("Precision@10%", f"{metrics.get('precision_10', 0):.3f}")
            st.metric("Decile-1 Lift", f"{metrics.get('lift_decile1', 0):.2f}×")

        st.divider()
        st.markdown("**API Endpoint**")
        st.code("POST http://localhost:5000/predict", language="text")

    # ── Main content ──────────────────────────────────────────────────────────
    st.markdown("# 📡 Telecom Churn Prediction Engine")
    st.caption("Real-time subscriber churn scoring · XGBoost + SHAP · Banglalink CVM Division")

    # Load data
    df = load_scored_data()

    if df is None:
        st.error("⚠️  No scored data found. Please run `make train` first to generate features and train the model.")
        st.code("make train", language="bash")
        st.stop()

    # Apply filters
    filtered = df.copy()
    if risk_filter != "ALL":
        filtered = filtered[filtered["risk_tier"] == risk_filter]
    if segment_filter:
        filtered = filtered[filtered["segment"].isin(segment_filter)]

    top_df = filtered.nlargest(top_n, "churn_prob").reset_index(drop=True)

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    total_subs  = len(df)
    at_risk     = (df["risk_tier"] == "HIGH").sum()
    med_risk    = (df["risk_tier"] == "MEDIUM").sum()
    avg_prob    = df[df["risk_tier"] == "HIGH"]["churn_prob"].mean()
    arpu_at_risk = df[df["risk_tier"] == "HIGH"]["arpu_last_3m"].mean()
    projected_loss_cr = (at_risk * arpu_at_risk * 12) / 1_00_00_000  # Crore BDT

    c1, c2, c3, c4, c5 = st.columns(5)
    def kpi(col, value, label, delta=None):
        delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
            {delta_html}
        </div>""", unsafe_allow_html=True)

    kpi(c1, f"{total_subs:,}", "Total Subscribers")
    kpi(c2, f"{at_risk:,}", "High-Risk Subscribers", "🔴 Immediate Attention")
    kpi(c3, f"{med_risk:,}", "Medium-Risk Subscribers", "🟡 Monitor Closely")
    kpi(c4, f"{avg_prob:.1%}", "Avg Churn Prob (HIGH)", "")
    kpi(c5, f"৳{projected_loss_cr:.1f} Cr", "Projected Annual Loss", "Est. revenue at risk")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column layout ─────────────────────────────────────────────────────
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        st.markdown('<div class="section-header">🎯 Top At-Risk Subscribers</div>',
                    unsafe_allow_html=True)

        # Risk tier badge
        def risk_badge(t):
            colors = {"HIGH": "#FF6B6B", "MEDIUM": "#FFB347", "LOW": "#4CAF50"}
            return f'<span style="color:{colors.get(t,"#aaa")};font-weight:700">{t}</span>'

        display_df = top_df[[
            "subscriber_id", "churn_prob", "risk_tier", "segment",
            "arpu_last_3m", "last_recharge_days", "support_tickets_90d"
        ]].copy()
        display_df["churn_prob"] = display_df["churn_prob"].map(lambda x: f"{x:.2%}")
        display_df.columns = [
            "Subscriber ID", "Churn Prob", "Risk Tier", "Segment",
            "ARPU (BDT)", "Recharge Gap (d)", "Support Tickets"
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            height=420,
            hide_index=True,
        )

        # Download
        csv = top_df.to_csv(index=False).encode()
        st.download_button(
            "⬇  Export Top-Risk CSV",
            data=csv,
            file_name=f"top_risk_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    with right_col:
        st.markdown('<div class="section-header">📊 Risk Distribution</div>',
                    unsafe_allow_html=True)

        tier_counts = df["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Risk Tier", "Count"]
        tier_colors = {"HIGH": "#FF6B6B", "MEDIUM": "#FFB347", "LOW": "#4CAF50"}
        tier_counts["Color"] = tier_counts["Risk Tier"].map(tier_colors)

        fig_donut = go.Figure(go.Pie(
            labels=tier_counts["Risk Tier"],
            values=tier_counts["Count"],
            hole=0.6,
            marker=dict(colors=tier_counts["Color"].tolist(),
                        line=dict(color="#0F1117", width=3)),
            textinfo="label+percent",
            textfont=dict(color="#E8EAF0", size=12),
        ))
        fig_donut.update_layout(
            paper_bgcolor="#0F1117", plot_bgcolor="#0F1117",
            font=dict(color="#E8EAF0"),
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            height=240,
            annotations=[dict(text=f"{total_subs:,}<br>Subscribers",
                             x=0.5, y=0.5, font_size=13,
                             font_color="#E8EAF0", showarrow=False)]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown('<div class="section-header">📉 Churn Probability Distribution</div>',
                    unsafe_allow_html=True)

        fig_hist = px.histogram(
            df, x="churn_prob", nbins=40,
            color_discrete_sequence=["#00C2FF"],
        )
        fig_hist.update_traces(opacity=0.8)
        fig_hist.update_layout(
            paper_bgcolor="#0F1117", plot_bgcolor="#1A1D27",
            font=dict(color="#E8EAF0"),
            xaxis=dict(title="Churn Probability", gridcolor="#2A2D3A"),
            yaxis=dict(title="Count", gridcolor="#2A2D3A"),
            margin=dict(l=40, r=20, t=20, b=40),
            height=220,
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # ── SHAP Explorer ─────────────────────────────────────────────────────────
    st.markdown("## 🔍 Subscriber SHAP Explorer")
    st.caption("Select a subscriber to see which features drive their churn score.")

    model, feat_names = load_model()
    if model and feat_names:
        sub_ids = top_df["subscriber_id"].tolist()
        selected = st.selectbox("Select Subscriber", sub_ids, index=0)

        row = top_df[top_df["subscriber_id"] == selected].iloc[0]
        X_row = row[feat_names].values.reshape(1, -1)
        sv = compute_shap_row(model, X_row, feat_names)

        info_col, chart_col = st.columns([1, 2])
        with info_col:
            prob = float(row["churn_prob"])
            tier = row["risk_tier"]
            tier_color = {"HIGH": "#FF6B6B", "MEDIUM": "#FFB347", "LOW": "#4CAF50"}[tier]
            st.markdown(f"""
            <div class="kpi-card">
                <div style="font-size:0.85rem;color:#8A8FAA;margin-bottom:6px">
                    {selected}
                </div>
                <div class="kpi-value">{prob:.1%}</div>
                <div style="color:{tier_color};font-weight:700;margin-top:4px">
                    {tier} RISK
                </div>
                <hr style="border-color:#2A2D3A;margin:12px 0">
                <div style="text-align:left;font-size:0.82rem;color:#8A8FAA">
                    Segment: <b style="color:#E8EAF0">{row.get("segment","N/A")}</b><br>
                    Tenure:  <b style="color:#E8EAF0">{int(row.get("contract_months",0))} mo</b><br>
                    ARPU:    <b style="color:#E8EAF0">৳{row.get("arpu_last_3m",0):.0f}</b><br>
                    Recharge Gap: <b style="color:#E8EAF0">{int(row.get("last_recharge_days",0))}d</b>
                </div>
                <hr style="border-color:#2A2D3A;margin:12px 0">
                <div style="font-size:0.8rem;color:#8A8FAA;text-align:left">
                    Recommended Action:<br>
                    <b style="color:#00C2FF">{RETENTION_ACTIONS[tier]}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with chart_col:
            st.plotly_chart(
                shap_waterfall_chart(sv, feat_names),
                use_container_width=True
            )

    # ── Segment analysis ──────────────────────────────────────────────────────
    st.divider()
    st.markdown("## 📊 Churn Risk by Segment")

    seg_stats = df.groupby("segment").agg(
        total=("churn_prob", "count"),
        avg_prob=("churn_prob", "mean"),
        high_risk=("risk_tier", lambda x: (x == "HIGH").sum()),
    ).reset_index()
    seg_stats["high_risk_pct"] = (seg_stats["high_risk"] / seg_stats["total"] * 100).round(1)

    fig_seg = go.Figure()
    fig_seg.add_trace(go.Bar(
        x=seg_stats["segment"], y=seg_stats["avg_prob"] * 100,
        name="Avg Churn Prob (%)",
        marker_color="#00C2FF", opacity=0.85,
    ))
    fig_seg.add_trace(go.Bar(
        x=seg_stats["segment"], y=seg_stats["high_risk_pct"],
        name="High-Risk %",
        marker_color="#FF6B6B", opacity=0.85,
    ))
    fig_seg.update_layout(
        barmode="group",
        paper_bgcolor="#0F1117", plot_bgcolor="#1A1D27",
        font=dict(color="#E8EAF0"),
        legend=dict(orientation="h", y=1.12),
        xaxis=dict(title="Segment", gridcolor="#2A2D3A"),
        yaxis=dict(title="%", gridcolor="#2A2D3A"),
        margin=dict(l=40, r=20, t=40, b=40),
        height=320,
    )
    st.plotly_chart(fig_seg, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "Telecom Churn Prediction Engine · XGBoost + SHAP · "
        f"Scored {total_subs:,} subscribers · "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')} · "
        "Banglalink CVM Division"
    )


if __name__ == "__main__":
    main()
