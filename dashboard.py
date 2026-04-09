"""
SIH1649 - DDoS Protection System
Professional Dashboard - Dark Theme
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import json
import warnings
warnings.filterwarnings('ignore')
from PIL import Image

st.set_page_config(
    page_title="DDoS Protection System | SIH1649",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── GLOBAL ── */
html, body, [class*="css"], .stApp {
    background-color: #0e1117 !important;
    color: #ffffff !important;
    font-family: 'Segoe UI', sans-serif;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background-color: #1a1f35 !important;
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* ── HEADER CARD ── */
.main-header {
    background: linear-gradient(135deg, #1a1f35, #0d1b2a);
    border: 2px solid #00d4ff;
    border-radius: 16px;
    padding: 30px;
    margin-bottom: 20px;
    text-align: center;
}
.main-header h1 {
    color: #00d4ff;
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0 0 8px 0;
}
.main-header p {
    color: #ccd6f6;
    font-size: 0.9rem;
    margin: 4px 0;
}

/* ── METRIC CARDS ── */
[data-testid="metric-container"] {
    background: #1a1f35 !important;
    border: 1px solid #00d4ff !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] p {
    color: #8892b0 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="stMetricValue"] {
    color: #00d4ff !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
}

/* ── TABS ── */
[data-baseweb="tab-list"] {
    background: #1a1f35 !important;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
[data-baseweb="tab"] {
    color: #8892b0 !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
}
[aria-selected="true"] {
    background: #0e1117 !important;
    color: #00d4ff !important;
}

/* ── SECTION TITLE ── */
.section-title {
    color: #ffffff;
    font-size: 1.05rem;
    font-weight: 700;
    border-left: 4px solid #00d4ff;
    padding-left: 12px;
    margin: 20px 0 12px 0;
}

/* ── ALERTS ── */
.alert-danger {
    background: rgba(255,71,87,0.15);
    border: 2px solid #ff4757;
    border-radius: 10px;
    padding: 18px;
    color: #ff6b78;
    font-size: 1.1rem;
    font-weight: 700;
    text-align: center;
}
.alert-safe {
    background: rgba(0,255,136,0.1);
    border: 2px solid #00ff88;
    border-radius: 10px;
    padding: 18px;
    color: #00ff88;
    font-size: 1.1rem;
    font-weight: 700;
    text-align: center;
}

/* ── SLIDERS ── */
[data-testid="stSlider"] label {
    color: #ccd6f6 !important;
    font-weight: 500 !important;
}
[data-testid="stSlider"] p {
    color: #00d4ff !important;
    font-weight: 600 !important;
}

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #0099cc) !important;
    color: #0e1117 !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00eeff, #00aadd) !important;
    transform: translateY(-1px);
}

/* ── TABLE ── */
table {
    color: #ffffff !important;
    width: 100%;
}
th {
    color: #00d4ff !important;
    background: #1a1f35 !important;
    padding: 8px 12px !important;
}
td {
    color: #ccd6f6 !important;
    padding: 6px 12px !important;
    border-bottom: 1px solid #2a3050 !important;
}

/* ── GENERAL TEXT ── */
p, li, span, div {
    color: #ccd6f6;
}
h2, h3, h4 {
    color: #ffffff !important;
}
strong {
    color: #ffffff !important;
}

/* ── SUCCESS/ERROR BOXES ── */
[data-testid="stAlert"] {
    background: #1a1f35 !important;
    border-radius: 8px !important;
}

/* ── DIVIDER ── */
hr {
    border-color: #2a3050 !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    background: #1a1f35 !important;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ──────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ Distributed Denial of Service (DDoS) Protection System for Cloud</h1>
    <p>Real-time Multi-class Attack Detection and Classification using Machine Learning</p>
    <p style="color:#8892b0; font-size:0.82rem;">
        B.Tech Final Year Project &nbsp;|&nbsp; Computer Science &amp; Engineering
        &nbsp;|&nbsp; SIH1649 &nbsp;|&nbsp; CIC-DDoS2019 Dataset
    </p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ─────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ System Info")
    st.markdown("**Dataset:** CIC-DDoS2019")
    st.markdown("**Algorithm:** Random Forest Classifier")
    st.markdown("**Features:** 82 network flow parameters")
    st.divider()
    st.markdown("**DDoS Attack Types:**")
    attacks = ["DrDoS_DNS","DrDoS_LDAP","DrDoS_MSSQL","DrDoS_NTP",
               "DrDoS_NetBIOS","DrDoS_SNMP","DrDoS_SSDP","DrDoS_UDP",
               "Syn Flood","TFTP","UDP-lag","BENIGN (Normal)"]
    for a in attacks:
        st.markdown(f"• {a}")
    st.divider()
    st.markdown("**What is DDoS?**")
    st.markdown("A Distributed Denial of Service attack floods a server with massive traffic, making it unavailable to users.")

# ── LOAD MODEL ──────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model         = joblib.load("model/ddos_model.pkl")
        scaler        = joblib.load("model/scaler.pkl")
        le            = joblib.load("model/label_encoder.pkl")
        feature_names = joblib.load("model/feature_names.pkl")
        with open("model/stats.json") as f:
            stats = json.load(f)
        return model, scaler, le, feature_names, stats, True
    except Exception as e:
        return None, None, None, None, None, False

model, scaler, le, feature_names, stats, model_loaded = load_model()

if not model_loaded:
    st.error("❌ Model files not found!")
    st.stop()

st.success("✅ Model loaded successfully!")

# ── METRICS ─────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("📦 Total Samples",      f"{stats['total_samples']:,}")
c2.metric("⚠️ DDoS Attack Types",  f"{stats['attack_types']}")
c3.metric("🎯 Detection Accuracy", f"{stats['accuracy']}%")
c4.metric("🔬 Network Features",   f"{stats['features']}")

st.divider()

# ── TABS ────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Live Detection",
    "📊 Data Analysis",
    "🎯 Model Performance",
    "📖 Project Overview"
])

# ════════════════════════════════════
# TAB 1 — Live Detection
# ════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Real-time DDoS Attack Classification</div>',
                unsafe_allow_html=True)
    st.markdown("Adjust the network traffic parameters below and click **Classify** to detect the attack type.")
    st.markdown("")

    importances  = model.feature_importances_
    top_features = pd.Series(importances, index=feature_names)\
                     .sort_values(ascending=False).head(8).index.tolist()

    inputs = {}
    cols = st.columns(4)
    for i, feat in enumerate(top_features):
        inputs[feat] = cols[i % 4].slider(
            feat[:24], 0.0, 1000000.0, 100.0, key=f"s_{feat}")

    st.markdown("")
    left_p, right_p = st.columns([1.2, 1])

    with left_p:
        predict_btn = st.button("🔍  Classify Traffic", use_container_width=True)

        if predict_btn:
            row = pd.DataFrame([{f: 0.0 for f in feature_names}])
            for feat, val in inputs.items():
                row[feat] = val
            row_sc = scaler.transform(row[feature_names])
            pred   = model.predict(row_sc)[0]
            proba  = model.predict_proba(row_sc)[0]
            label  = le.inverse_transform([pred])[0]
            conf   = proba[pred] * 100

            st.markdown("")
            if label == "BENIGN":
                st.markdown(
                    f'<div class="alert-safe">'
                    f'✅ BENIGN — Normal Traffic Detected<br>'
                    f'<span style="font-size:0.88rem;font-weight:400">'
                    f'Confidence: {conf:.1f}% &nbsp;|&nbsp; No threat identified'
                    f'</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="alert-danger">'
                    f'🚨 DDoS ATTACK DETECTED: {label}<br>'
                    f'<span style="font-size:0.88rem;font-weight:400">'
                    f'Confidence: {conf:.1f}% &nbsp;|&nbsp; Immediate mitigation required'
                    f'</span></div>', unsafe_allow_html=True)

            st.markdown("#### 📊 Classification Probabilities")
            top3 = np.argsort(proba)[::-1][:3]
            for idx in top3:
                name  = le.inverse_transform([idx])[0]
                pct   = proba[idx] * 100
                color = "#00ff88" if name == "BENIGN" else "#ff4757"
                bar   = int(pct / 3)
                st.markdown(
                    f"<span style='color:#ffffff;font-weight:600'>{name}</span>"
                    f"&nbsp;&nbsp;<code style='background:#1a1f35;color:#00d4ff'>{pct:.1f}%</code>"
                    f"&nbsp;<span style='color:{color}'>{'█' * bar}</span>",
                    unsafe_allow_html=True)
                st.markdown("")

    with right_p:
        st.markdown("#### 📋 Feature Reference")
        st.markdown("""
| Parameter | Meaning |
|---|---|
| Flow Duration | Connection length |
| Pkt Length Mean | Average packet size |
| Fwd Packets/s | Packets per second |
| SYN Flag Count | Connection requests |
| Bwd IAT Mean | Time between packets |
""")
        st.markdown("")
        

# ════════════════════════════════════
# TAB 2 — Data Analysis
# ════════════════════════════════════
with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Attack Type Distribution</div>', unsafe_allow_html=True)
        if os.path.exists("attack_distribution.png"):
            st.image(Image.open("attack_distribution.png"), use_container_width=True)
        else:
            st.warning("attack_distribution.png not found")

    with col_b:
        st.markdown('<div class="section-title">Top 10 Important Features</div>', unsafe_allow_html=True)
        if os.path.exists("feature_importance.png"):
            st.image(Image.open("feature_importance.png"), use_container_width=True)
        else:
            st.warning("feature_importance.png not found")

# ════════════════════════════════════
# TAB 3 — Model Performance
# ════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Confusion Matrix — Predicted vs Actual</div>',
                unsafe_allow_html=True)
    if os.path.exists("confusion_matrix.png"):
        st.image(Image.open("confusion_matrix.png"),
                 caption=f"Overall Accuracy: {stats['accuracy']}%",
                 use_container_width=True)
    else:
        st.warning("confusion_matrix.png not found")

    st.markdown('<div class="section-title">Model Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",     f"{stats['accuracy']}%")
    c2.metric("Samples",      f"{stats['total_samples']:,}")
    c3.metric("Attack Types", f"{stats['attack_types']}")
    c4.metric("Features",     f"{stats['features']}")

# ════════════════════════════════════
# TAB 4 — Project Overview
# ════════════════════════════════════
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📌 Problem Statement")
        st.markdown("""
**Distributed Denial of Service (DDoS)** attacks are one of the most critical
threats to cloud infrastructure. Attackers flood servers with massive traffic
from multiple compromised sources simultaneously — making services unavailable
to legitimate users.

**This project implements:**
- Real-time network traffic analysis
- Classification into 11 DDoS attack types
- Instant threat alerts with confidence scores
- Interactive web dashboard for monitoring
        """)

        st.markdown("### 🔬 Methodology")
        st.markdown("""
1. **Data Collection** — CIC-DDoS2019 (Canadian Institute for Cybersecurity)
2. **Preprocessing** — Null removal, infinity handling, feature scaling
3. **Feature Extraction** — 82 network flow parameters
4. **Model Training** — Random Forest Classifier (100 estimators)
5. **Evaluation** — Confusion matrix, Precision, Recall, F1-score
6. **Deployment** — Streamlit Cloud dashboard
        """)

    with col2:
        st.markdown("### 📊 Results Summary")
        st.markdown(f"""
| Metric | Value |
|---|---|
| Overall Accuracy | **{stats['accuracy']}%** |
| Total Samples | **{stats['total_samples']:,}** |
| Attack Types | **{stats['attack_types']}** |
| Features Used | **{stats['features']}** |
| Algorithm | **Random Forest** |
| Dataset | **CIC-DDoS2019** |
        """)

        st.markdown("### ⚙️ Tech Stack")
        st.markdown("""
| Component | Technology |
|---|---|
| Language | Python 3.12 |
| ML Library | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit Cloud |
| Model Storage | Joblib |
        """)

# ── FOOTER ──────────────────────────────────────
st.divider()
st.markdown(
    "<center style='color:#4a5568;font-size:0.8rem'>"
    "Distributed Denial of Service (DDoS) Protection System for Cloud &nbsp;|&nbsp;"
    " B.Tech Final Year Project &nbsp;|&nbsp; SIH1649 &nbsp;|&nbsp; CIC-DDoS2019"
    "</center>", unsafe_allow_html=True)
