"""
SIH1649 - DDoS Protection System
Fixed UI — Light theme compatible
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
    page_title="DDoS Protection System | B.Tech Final Year Project",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Force white background everywhere */
    .stApp { background-color: #f0f2f6 !important; }
    section[data-testid="stSidebar"] { background-color: #1a1f35 !important; }
    section[data-testid="stSidebar"] * { color: #ffffff !important; }

    .main-header {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1b2a 100%);
        border: 2px solid #00d4ff;
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 16px;
        text-align: center;
    }
    .main-header h1 {
        color: #00d4ff !important;
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0 0 6px 0;
    }
    .main-header .subtitle {
        color: #ffffff !important;
        font-size: 0.95rem;
        margin: 4px 0;
    }
    .main-header .meta {
        color: #aabbcc !important;
        font-size: 0.82rem;
        margin: 10px 0 0 0;
    }

    .section-title {
        color: #1a1f35 !important;
        font-size: 1.1rem;
        font-weight: 700;
        border-left: 4px solid #00d4ff;
        padding-left: 12px;
        margin: 20px 0 14px 0;
        background: transparent;
    }

    .alert-danger {
        background: #fff0f0;
        border: 2px solid #ff4757;
        border-radius: 10px;
        padding: 16px 20px;
        color: #cc0000 !important;
        font-size: 1.1rem;
        font-weight: 700;
        text-align: center;
    }
    .alert-safe {
        background: #f0fff4;
        border: 2px solid #00aa44;
        border-radius: 10px;
        padding: 16px 20px;
        color: #006622 !important;
        font-size: 1.1rem;
        font-weight: 700;
        text-align: center;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1a1f35 !important;
        border: 2px solid #00d4ff !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }
    [data-testid="metric-container"] label {
        color: #aabbcc !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        color: #1a1f35 !important;
        font-weight: 600;
        font-size: 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: #0d1b2a !important;
        border-bottom: 3px solid #00d4ff !important;
    }

    /* General text */
    p, li, label { color: #1a1f35 !important; }
    h1, h2, h3, h4 { color: #1a1f35 !important; }

    /* Probability bars */
    .prob-bar-green {
        background: #00aa44;
        height: 20px;
        border-radius: 4px;
        display: inline-block;
    }
    .prob-bar-red {
        background: #ff4757;
        height: 20px;
        border-radius: 4px;
        display: inline-block;
    }

    /* Button */
    .stButton button {
        background: #1a1f35 !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: 2px solid #00d4ff !important;
        font-size: 1rem !important;
    }
    .stButton button:hover {
        background: #00d4ff !important;
        color: #1a1f35 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
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

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ System Info")
    st.markdown("**Dataset:** CIC-DDoS2019")
    st.markdown("**Algorithm:** Random Forest Classifier")
    st.markdown("**Features:** 82 network flow parameters")
    st.divider()
    st.markdown("**DDoS Attack Types:**")
    for a in ["DrDoS_DNS","DrDoS_LDAP","DrDoS_MSSQL","DrDoS_NTP",
              "DrDoS_NetBIOS","DrDoS_SNMP","DrDoS_SSDP","DrDoS_UDP",
              "Syn Flood","TFTP","UDP-lag","BENIGN (Normal)"]:
        st.markdown(f"- {a}")
    st.divider()
    st.markdown("""
    **What is DDoS?**
    A Distributed Denial of Service attack floods a server with traffic from multiple sources,
    making it unavailable to legitimate users.
    """)

# Load model
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
    st.error("❌ Model files not found! Please run save_model.py first.")
    st.stop()

st.success("✅ Model loaded successfully!")

# Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("📦 Total Samples",      f"{stats['total_samples']:,}")
c2.metric("⚠️ DDoS Attack Types",  f"{stats['attack_types']}")
c3.metric("🎯 Detection Accuracy", f"{stats['accuracy']}%")
c4.metric("🔬 Network Features",   f"{stats['features']}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Live Detection",
    "📊 Data Analysis",
    "🎯 Model Performance",
    "📖 Project Overview"
])

# TAB 1
with tab1:
    st.markdown('<div class="section-title">Real-time DDoS Attack Classification</div>',
                unsafe_allow_html=True)
    st.write("Adjust network traffic parameters below and click **Classify** to detect attack type.")

    importances  = model.feature_importances_
    top_features = pd.Series(importances, index=feature_names)\
                     .sort_values(ascending=False).head(8).index.tolist()

    inputs = {}
    cols = st.columns(4)
    for i, feat in enumerate(top_features):
        inputs[feat] = cols[i % 4].slider(
            feat[:22], 0.0, 1000000.0, 100.0, key=f"s_{feat}")

    st.markdown("")
    left_p, right_p = st.columns([1, 1])

    with left_p:
        if st.button("🔍 Classify Traffic", use_container_width=True):
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
                    f'<small>Confidence: {conf:.1f}% &nbsp;|&nbsp; No threat identified</small></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="alert-danger">🚨 DDoS ATTACK DETECTED: {label}<br>'
                    f'<small>Confidence: {conf:.1f}% &nbsp;|&nbsp; Immediate mitigation required</small></div>',
                    unsafe_allow_html=True)

            st.markdown("#### Classification Probabilities")
            for idx in np.argsort(proba)[::-1][:3]:
                name = le.inverse_transform([idx])[0]
                pct  = proba[idx] * 100
                bar  = int(pct / 2)
                color = "green" if name == "BENIGN" else "red"
                st.markdown(
                    f"**{name}** — `{pct:.1f}%` "
                    f"<span style='color:{color}'>{'█' * bar}</span>",
                    unsafe_allow_html=True)

    with right_p:
        st.markdown("#### 📋 Feature Reference")
        st.markdown("""
        | Parameter | Meaning |
        |---|---|
        | Flow Duration | Connection length |
        | Packet Length Mean | Average packet size |
        | Fwd Packets/s | Packets sent per second |
        | SYN Flag Count | Connection request count |
        | Bwd IAT Mean | Time between incoming packets |

        ---
        **Demo Tip:**
        Set all sliders to **maximum** and click Classify — watch the attack get detected! 🚨
        """)

# TAB 2
with tab2:
    st.markdown('<div class="section-title">DDoS Attack Type Distribution</div>',
                unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        if os.path.exists("attack_distribution.png"):
            st.image(Image.open("attack_distribution.png"),
                     caption="Attack Distribution — CIC-DDoS2019",
                     use_container_width=True)
        else:
            st.info("attack_distribution.png not found in project folder")
    with col_b:
        if os.path.exists("feature_importance.png"):
            st.image(Image.open("feature_importance.png"),
                     caption="Top Features — Random Forest Model",
                     use_container_width=True)
        else:
            st.info("feature_importance.png not found in project folder")

# TAB 3
with tab3:
    st.markdown('<div class="section-title">Confusion Matrix — Predicted vs Actual</div>',
                unsafe_allow_html=True)
    if os.path.exists("confusion_matrix.png"):
        st.image(Image.open("confusion_matrix.png"),
                 caption=f"Confusion Matrix | Overall Accuracy: {stats['accuracy']}%",
                 use_container_width=True)
    else:
        st.info("confusion_matrix.png not found")

    st.markdown('<div class="section-title">Model Summary</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Accuracy", f"{stats['accuracy']}%")
    c2.metric("Total Samples",    f"{stats['total_samples']:,}")
    c3.metric("Attack Classes",   f"{stats['attack_types']}")

# TAB 4
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📌 Problem Statement")
        st.markdown("""
        **Distributed Denial of Service (DDoS)** attacks are one of the most critical threats
        to cloud infrastructure. Attackers flood servers with massive traffic from multiple
        sources, making services unavailable to legitimate users.

        This project implements:
        - Real-time network traffic analysis
        - Classification into 11 DDoS attack types
        - Instant threat alerts
        - Interactive web dashboard
        """)
        st.markdown("### 🔬 Methodology")
        st.markdown("""
        1. **Data Collection** — CIC-DDoS2019 (Canadian Institute for Cybersecurity)
        2. **Preprocessing** — Null removal, feature scaling
        3. **Feature Extraction** — 82 network flow parameters
        4. **Model Training** — Random Forest (100 estimators)
        5. **Evaluation** — Confusion matrix, F1-score
        6. **Deployment** — Streamlit web dashboard
        """)
    with col2:
        st.markdown("### 📊 Results")
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
        | Data | Pandas, NumPy |
        | Visualization | Matplotlib, Seaborn |
        | Dashboard | Streamlit |
        | Storage | Joblib |
        """)

st.divider()
st.markdown(
    "<center style='color:#555;font-size:0.85rem;'>"
    "Distributed Denial of Service (DDoS) Protection System for Cloud &nbsp;|&nbsp;"
    " B.Tech Final Year Project &nbsp;|&nbsp; SIH1649 &nbsp;|&nbsp; CIC-DDoS2019"
    "</center>", unsafe_allow_html=True)
