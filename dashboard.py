"""
SIH1649 - DDoS Protection System
Cloud-compatible Dashboard — loads from saved model only
No CSV files needed!
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.set_page_config(
    page_title="DDoS Protection System | B.Tech Final Year Project",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main-header {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1b2a 100%);
        border: 1px solid #00d4ff33;
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 8px;
        text-align: center;
    }
    .main-header h1 { color: #00d4ff; font-size: 2.1rem; font-weight: 800; margin: 0 0 6px 0; }
    .main-header .subtitle { color: #ccd6f6; font-size: 0.95rem; margin: 4px 0; }
    .main-header .meta { color: #8892b0; font-size: 0.82rem; margin: 10px 0 0 0; }
    .section-title {
        color: #ccd6f6; font-size: 1.1rem; font-weight: 700;
        border-left: 4px solid #00d4ff;
        padding-left: 12px; margin: 20px 0 14px 0;
    }
    .alert-danger {
        background: #ff475718; border: 1px solid #ff4757;
        border-radius: 10px; padding: 16px 20px;
        color: #ff4757; font-size: 1.1rem; font-weight: 700; text-align: center;
    }
    .alert-safe {
        background: #00ff8818; border: 1px solid #00ff88;
        border-radius: 10px; padding: 16px 20px;
        color: #00ff88; font-size: 1.1rem; font-weight: 700; text-align: center;
    }
    [data-testid="metric-container"] {
        background: #1a1f35; border: 1px solid #00d4ff22;
        border-radius: 12px; padding: 16px;
    }
    [data-testid="metric-container"] label {
        color: #8892b0 !important; font-size: 0.8rem !important;
        text-transform: uppercase; letter-spacing: 1px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #00d4ff !important; font-size: 1.8rem !important; font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🛡️ Distributed Denial of Service (DDoS) Protection System for Cloud</h1>
    <p class="subtitle">Real-time Multi-class Attack Detection and Classification using Machine Learning</p>
    <p class="meta">
        B.Tech Final Year Project &nbsp;|&nbsp; Computer Science &amp; Engineering
        &nbsp;|&nbsp; Problem Statement: SIH1649
        &nbsp;|&nbsp; Dataset: CIC-DDoS2019 (Canadian Institute for Cybersecurity)
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## System Info")
    st.markdown("**Dataset:** CIC-DDoS2019")
    st.markdown("**Algorithm:** Random Forest Classifier")
    st.markdown("**Features:** 82 network flow parameters")
    st.divider()
    st.markdown("**DDoS Attack Types Covered:**")
    for a in ["DrDoS_DNS","DrDoS_LDAP","DrDoS_MSSQL","DrDoS_NTP",
              "DrDoS_NetBIOS","DrDoS_SNMP","DrDoS_SSDP","DrDoS_UDP",
              "Syn Flood","TFTP","UDP-lag","BENIGN (Normal)"]:
        st.markdown(f"- `{a}`")
    st.divider()
    st.markdown("""
    **What is DDoS?**
    A **Distributed Denial of Service** attack floods a
    server with traffic from multiple sources, making it
    unavailable to legitimate users.
    """)

# Load saved model only
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
    st.error("Model files not found! Please run save_model.py first.")
    st.stop()

st.success("✅ Model loaded successfully!")

# Metrics from stats.json
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Samples",      f"{stats['total_samples']:,}")
c2.metric("DDoS Attack Types",  f"{stats['attack_types']}")
c3.metric("Detection Accuracy", f"{stats['accuracy']}%")
c4.metric("Network Features",   f"{stats['features']}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Live Detection", "📊 Data Analysis", "🎯 Model Performance", "📖 Project Overview"
])

# TAB 1 - Live Detection
with tab1:
    st.markdown('<div class="section-title">Real-time DDoS attack classification</div>', unsafe_allow_html=True)
    st.caption("Adjust network traffic parameters and click Classify to detect attack type.")

    # Use feature importance to show top sliders
    importances  = model.feature_importances_
    top_features = pd.Series(importances, index=feature_names)\
                     .sort_values(ascending=False).head(8).index.tolist()

    # Build reference ranges from feature importance
    feature_ranges = {
        feat: {"min": 0.0, "max": 1000000.0, "default": 100.0}
        for feat in top_features
    }

    inputs = {}
    cols = st.columns(4)
    for i, feat in enumerate(top_features):
        r = feature_ranges[feat]
        inputs[feat] = cols[i % 4].slider(
            feat[:22], r["min"], r["max"], r["default"], key=f"s_{feat}")

    st.markdown("")
    left_p, right_p = st.columns([1,1])

    with left_p:
        if st.button("🔍 Classify Traffic", use_container_width=True, type="primary"):
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
                    f'<div class="alert-safe">✅ BENIGN — Normal Traffic Detected<br>'
                    f'<span style="font-size:0.88rem;font-weight:400">'
                    f'Confidence: {conf:.1f}% | No threat identified</span></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="alert-danger">🚨 DDoS ATTACK DETECTED: {label}<br>'
                    f'<span style="font-size:0.88rem;font-weight:400">'
                    f'Confidence: {conf:.1f}% | Immediate mitigation required</span></div>',
                    unsafe_allow_html=True)

            st.markdown("#### Classification Probabilities")
            for idx in np.argsort(proba)[::-1][:3]:
                name  = le.inverse_transform([idx])[0]
                pct   = proba[idx] * 100
                color = "#00ff88" if name == "BENIGN" else "#ff4757"
                bar   = int(pct / 4)
                st.markdown(
                    f"**{name}** `{pct:.1f}%` "
                    f"<span style='color:{color}'>{'█' * bar}</span>",
                    unsafe_allow_html=True)

    with right_p:
        st.markdown("#### Feature Reference")
        st.markdown("""
        | Parameter | Meaning |
        |---|---|
        | Flow Duration | Connection length |
        | Packet Length | Size of data packets |
        | Fwd Packets/s | Packets sent per second |
        | SYN Flag Count | Connection request count |
        | Bwd IAT Mean | Time between incoming packets |

        ---
        **Demo tip:** Set any value to maximum
        and click Classify to see attack detection!
        """)

# TAB 2 - Data Analysis (using saved PNG images)
with tab2:
    st.markdown('<div class="section-title">DDoS attack type distribution</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if os.path.exists("attack_distribution.png"):
            img = Image.open("attack_distribution.png")
            st.image(img, caption="DDoS Attack Type Distribution", use_container_width=True)
        else:
            st.info("attack_distribution.png not found")

    with col_b:
        if os.path.exists("feature_importance.png"):
            img = Image.open("feature_importance.png")
            st.image(img, caption="Top Features — Random Forest", use_container_width=True)
        else:
            st.info("feature_importance.png not found")

# TAB 3 - Model Performance
with tab3:
    st.markdown('<div class="section-title">Confusion matrix</div>', unsafe_allow_html=True)

    if os.path.exists("confusion_matrix.png"):
        img = Image.open("confusion_matrix.png")
        st.image(img, caption=f"Confusion Matrix | Accuracy: {stats['accuracy']}%",
                 use_container_width=True)
    else:
        st.info("confusion_matrix.png not found")

    st.markdown('<div class="section-title">Model summary</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Accuracy", f"{stats['accuracy']}%")
    col2.metric("Total Samples",    f"{stats['total_samples']:,}")
    col3.metric("Attack Classes",   f"{stats['attack_types']}")

    st.markdown("**Attack classes detected:**")
    classes = stats.get('classes', [])
    cols = st.columns(4)
    for i, c in enumerate(classes):
        cols[i % 4].markdown(f"- `{c}`")

# TAB 4 - Project Overview
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Problem Statement")
        st.markdown("""
        **Distributed Denial of Service (DDoS)** attacks are one of the most
        critical threats to cloud infrastructure. Attackers flood servers with
        massive traffic from multiple sources simultaneously, making services
        unavailable to legitimate users.

        This project proposes an **ML-based real-time detection system** that:
        - Analyses live network traffic flow features
        - Classifies traffic into 11 DDoS attack types
        - Provides instant alerts for detected attacks
        - Presents results through an interactive web dashboard
        """)
        st.markdown("### Methodology")
        st.markdown("""
        1. **Data Collection** — CIC-DDoS2019 dataset (Canadian Institute for Cybersecurity)
        2. **Preprocessing** — Null removal, infinity handling, feature scaling
        3. **Feature Extraction** — 82 network flow parameters
        4. **Model Training** — Random Forest Classifier (100 estimators)
        5. **Evaluation** — Confusion matrix, precision, recall, F1-score
        6. **Deployment** — Real-time web dashboard using Streamlit
        """)
    with col2:
        st.markdown("### Results Summary")
        st.markdown(f"""
        | Metric | Value |
        |---|---|
        | Overall Accuracy | **{stats['accuracy']}%** |
        | Total Samples | **{stats['total_samples']:,}** |
        | Attack Types Classified | **{stats['attack_types']}** |
        | Features Used | **{stats['features']}** |
        | Algorithm | **Random Forest** |
        | Dataset | **CIC-DDoS2019** |
        """)
        st.markdown("### Tech Stack")
        st.markdown("""
        | Layer | Technology |
        |---|---|
        | Language | Python 3.12 |
        | ML Library | Scikit-learn |
        | Data Processing | Pandas, NumPy |
        | Visualization | Matplotlib, Seaborn |
        | Dashboard | Streamlit |
        | Model Storage | Joblib |
        """)

st.divider()
st.markdown(
    "<center style='color:#4a5568;font-size:0.8rem'>"
    "Distributed Denial of Service (DDoS) Protection System for Cloud &nbsp;|&nbsp;"
    " B.Tech Final Year Project &nbsp;|&nbsp; SIH1649 &nbsp;|&nbsp; CIC-DDoS2019"
    "</center>", unsafe_allow_html=True)
