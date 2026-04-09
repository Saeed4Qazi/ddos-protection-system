"""
SIH1649 - DDoS Protection System
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, joblib, json, warnings
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
html, body, [class*="css"], .stApp {
    background-color: #0e1117 !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] { background-color: #1a1f35 !important; }
[data-testid="stSidebar"] * { color: #ffffff !important; }

.main-header {
    background: linear-gradient(135deg, #1a1f35, #0d1b2a);
    border: 2px solid #00d4ff;
    border-radius: 16px;
    padding: 30px;
    margin-bottom: 20px;
    text-align: center;
}
.main-header h1 { color: #00d4ff; font-size: 1.8rem; font-weight: 800; margin: 0 0 8px 0; }
.main-header p { color: #ccd6f6; font-size: 0.9rem; margin: 4px 0; }

.attack-card {
    background: #1a1f35;
    border: 1px solid #00d4ff33;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 10px;
}
.attack-card h4 { color: #00d4ff; margin: 0 0 6px 0; font-size: 0.95rem; }
.attack-card p { color: #ccd6f6; margin: 0; font-size: 0.82rem; line-height: 1.5; }

.model-card {
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 10px;
}
.model-a {
    background: linear-gradient(135deg, #1a3a1a, #0d2a0d);
    border: 2px solid #00ff88;
}
.model-b {
    background: linear-gradient(135deg, #1a2a3a, #0d1b2a);
    border: 2px solid #00d4ff;
}
.model-card h3 { margin: 0 0 8px 0; font-size: 1rem; }
.model-card .acc { font-size: 2.5rem; font-weight: 800; margin: 8px 0; }
.model-card p { color: #8892b0; font-size: 0.8rem; margin: 4px 0; }

.section-title {
    color: #ffffff; font-size: 1.05rem; font-weight: 700;
    border-left: 4px solid #00d4ff;
    padding-left: 12px; margin: 20px 0 12px 0;
}
.alert-danger {
    background: rgba(255,71,87,0.15); border: 2px solid #ff4757;
    border-radius: 10px; padding: 18px; color: #ff6b78;
    font-size: 1.1rem; font-weight: 700; text-align: center;
}
.alert-safe {
    background: rgba(0,255,136,0.1); border: 2px solid #00ff88;
    border-radius: 10px; padding: 18px; color: #00ff88;
    font-size: 1.1rem; font-weight: 700; text-align: center;
}
[data-testid="metric-container"] {
    background: #1a1f35 !important; border: 1px solid #00d4ff !important;
    border-radius: 12px !important; padding: 16px !important;
}
[data-testid="stMetricLabel"] p { color: #8892b0 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 1.8rem !important; font-weight: 800 !important; }
[data-baseweb="tab-list"] { background: #1a1f35 !important; border-radius: 8px; padding: 4px; }
[data-baseweb="tab"] { color: #8892b0 !important; font-weight: 600 !important; border-radius: 6px !important; }
[aria-selected="true"] { background: #0e1117 !important; color: #00d4ff !important; }
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #0099cc) !important;
    color: #0e1117 !important; font-weight: 800 !important;
    font-size: 1rem !important; border: none !important;
    border-radius: 10px !important; width: 100%;
}
[data-testid="stSlider"] label { color: #ccd6f6 !important; font-weight: 500 !important; }
p, li, span { color: #ccd6f6; }
h2, h3, h4 { color: #ffffff !important; }
strong { color: #ffffff !important; }
hr { border-color: #2a3050 !important; }
table { color: #ffffff !important; width: 100%; }
th { color: #00d4ff !important; background: #1a1f35 !important; padding: 8px 12px !important; }
td { color: #ccd6f6 !important; padding: 6px 12px !important; border-bottom: 1px solid #2a3050 !important; }
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="main-header">
    <h1>🛡️ Distributed Denial of Service (DDoS) Protection System for Cloud</h1>
    <p>Real-time Multi-class DDoS Attack Detection and Classification using Machine Learning</p>
    <p style="color:#8892b0;font-size:0.82rem;">
        Problem Statement: SIH1649 &nbsp;|&nbsp; Dataset: CIC-DDoS2019 (Canadian Institute for Cybersecurity)
    </p>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("## ⚙️ About This System")
    st.markdown("""
    This system uses **Machine Learning** to detect and classify
    Distributed Denial of Service (DDoS) attacks in real-time
    on cloud infrastructure.
    """)
    st.divider()
    st.markdown("**Dataset:** CIC-DDoS2019")
    st.markdown("**Algorithm:** Random Forest Classifier")
    st.markdown("**Total Features:** 82 network flow parameters")
    st.markdown("**Attack Types Covered:** 11 + BENIGN")
    st.divider()
    st.markdown("**Model Comparison:**")
    st.markdown("🟢 Model A — `01-12` only → **92.59%**")
    st.markdown("🔵 Model B — Both folders → **83.96%**")
    st.divider()
    st.markdown("**Problem Statement:** SIH1649")
    st.markdown("**Dataset Source:** [CIC-DDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html)")

# LOAD MODEL
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
    except:
        return None, None, None, None, None, False

model, scaler, le, feature_names, stats, model_loaded = load_model()
if not model_loaded:
    st.error("❌ Model files not found!")
    st.stop()

st.success("✅ Model loaded successfully!")

# METRICS
c1, c2, c3, c4 = st.columns(4)
c1.metric("📦 Total Samples",      f"{stats['total_samples']:,}")
c2.metric("⚠️ DDoS Attack Types",  f"{stats['attack_types']}")
c3.metric("🎯 Detection Accuracy", f"{stats['accuracy']}%")
c4.metric("🔬 Network Features",   f"{stats['features']}")

st.divider()

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚡ Live Detection",
    "🦠 Attack Types",
    "📊 Data Analysis",
    "🎯 Model Performance",
    "📖 Project Overview"
])

# ══════════════════════════════════
# TAB 1 — Live Detection
# ══════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Real-time DDoS Attack Classification</div>', unsafe_allow_html=True)
    st.markdown("Adjust the network traffic parameters below and click **Classify** to detect the attack type.")

    importances  = model.feature_importances_
    top_features = pd.Series(importances, index=feature_names)\
                     .sort_values(ascending=False).head(8).index.tolist()

    inputs = {}
    cols = st.columns(4)
    for i, feat in enumerate(top_features):
        inputs[feat] = cols[i % 4].slider(feat[:24], 0.0, 1000000.0, 100.0, key=f"s_{feat}")

    st.markdown("")
    left_p, right_p = st.columns([1.2, 1])

    with left_p:
        if st.button("🔍  Classify Traffic", use_container_width=True):
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
                    f'<span style="font-size:0.88rem;font-weight:400">Confidence: {conf:.1f}% | No threat identified</span></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="alert-danger">🚨 DDoS ATTACK DETECTED: {label}<br>'
                    f'<span style="font-size:0.88rem;font-weight:400">Confidence: {conf:.1f}% | Immediate mitigation required</span></div>',
                    unsafe_allow_html=True)

            st.markdown("#### 📊 Classification Probabilities")
            for idx in np.argsort(proba)[::-1][:3]:
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
        st.markdown("#### 📋 Parameter Reference")
        st.markdown("""
| Parameter | What it means |
|---|---|
| Flow Duration | How long connection lasted |
| Pkt Length Mean | Average size of packets |
| Fwd Packets/s | Packets sent per second |
| SYN Flag Count | No. of connection requests |
| Bwd IAT Mean | Gap between incoming packets |
""")

# ══════════════════════════════════
# TAB 2 — Attack Types
# ══════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">11 DDoS Attack Types Detected by This System</div>', unsafe_allow_html=True)

    attacks = [
        ("DrDoS_DNS", "DNS Amplification Attack",
         "Attacker sends small DNS queries with spoofed source IP (victim's IP). DNS server replies with huge responses to the victim — up to 40x amplification. Floods victim's bandwidth."),
        ("DrDoS_LDAP", "LDAP Amplification Attack",
         "Exploits LDAP servers to amplify traffic. Attacker sends small requests, LDAP server sends huge directory responses to the victim — up to 1000x amplification."),
        ("DrDoS_MSSQL", "Microsoft SQL Amplification Attack",
         "Targets Microsoft SQL Server's UDP port 1434. A small 'ping' packet causes a large response containing server info — used to flood victim with amplified traffic."),
        ("DrDoS_NTP", "NTP Amplification Attack",
         "Exploits Network Time Protocol servers using the MONLIST command. Returns list of last 600 clients — up to 556x amplification. One of the most powerful DDoS vectors."),
        ("DrDoS_NetBIOS", "NetBIOS Amplification Attack",
         "Exploits NetBIOS Name Service. Attacker sends broadcast queries, multiple machines respond to victim simultaneously. Common in older Windows-based corporate networks."),
        ("DrDoS_SNMP", "SNMP Amplification Attack",
         "Abuses Simple Network Management Protocol used to monitor network devices. GetBulk requests cause massive responses — up to 650x amplification. Most dangerous amplification attack."),
        ("DrDoS_SSDP", "SSDP Amplification Attack",
         "Targets Simple Service Discovery Protocol used by IoT/smart devices (TVs, cameras, Echo). Devices are tricked into sending large responses to victim — 30x amplification."),
        ("DrDoS_UDP", "UDP Flood Attack",
         "Sends massive amounts of UDP packets to random ports on victim. Server checks each packet, finds no application listening, sends ICMP 'unreachable' replies — exhausting resources."),
        ("Syn", "SYN Flood Attack",
         "Exploits TCP 3-way handshake. Attacker sends thousands of SYN (connection request) packets but never completes the handshake. Server holds open half-connections until memory is full."),
        ("TFTP", "TFTP Amplification Attack",
         "Exploits Trivial File Transfer Protocol — a simple, unauthenticated protocol used to transfer firmware to network devices. Large file transfers are directed to victim's IP."),
        ("UDP-lag", "UDP Lag Attack",
         "A variation of UDP flood designed to cause severe latency (lag) rather than a full crash. Specially crafted packets slow down processing — services become too slow to use."),
    ]

    col1, col2 = st.columns(2)
    for i, (attack_id, name, desc) in enumerate(attacks):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(f"""
<div class="attack-card">
    <h4>🔴 {attack_id} — {name}</h4>
    <p>{desc}</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("""
<div class="attack-card" style="border-color:#00ff8855;">
    <h4 style="color:#00ff88;">✅ BENIGN — Normal Traffic</h4>
    <p>Legitimate network traffic from real users — browsing, streaming, emails, API calls.
    The system correctly identifies this as safe and allows it through without blocking.</p>
</div>
""", unsafe_allow_html=True)

    # Model Accuracy Comparison
    st.markdown('<div class="section-title">Model Accuracy Comparison</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
<div class="model-card model-a">
    <h3 style="color:#00ff88;">🟢 Model A</h3>
    <div class="acc" style="color:#00ff88;">92.59%</div>
    <p>Trained on: <strong style="color:#fff">01-12 folder only</strong></p>
    <p>Total samples: <strong style="color:#fff">53,549</strong></p>
    <p>Attack types: <strong style="color:#fff">11 + BENIGN</strong></p>
    <p>Algorithm: <strong style="color:#fff">Random Forest (100 trees)</strong></p>
    <p style="color:#00ff88;margin-top:8px;">✅ Higher accuracy — cleaner dataset</p>
</div>
""", unsafe_allow_html=True)

    with col_b:
        st.markdown("""
<div class="model-card model-b">
    <h3 style="color:#00d4ff;">🔵 Model B</h3>
    <div class="acc" style="color:#00d4ff;">83.96%</div>
    <p>Trained on: <strong style="color:#fff">01-12 + 03-11 folders</strong></p>
    <p>Total samples: <strong style="color:#fff">87,536</strong></p>
    <p>Attack types: <strong style="color:#fff">12 + BENIGN</strong></p>
    <p>Algorithm: <strong style="color:#fff">Random Forest (100 trees)</strong></p>
    <p style="color:#00d4ff;margin-top:8px;">🔵 More data — better generalization</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════
# TAB 3 — Data Analysis
# ══════════════════════════════════
with tab3:
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

# ══════════════════════════════════
# TAB 4 — Model Performance
# ══════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Confusion Matrix — Predicted vs Actual</div>', unsafe_allow_html=True)
    if os.path.exists("confusion_matrix.png"):
        st.image(Image.open("confusion_matrix.png"),
                 caption=f"Overall Accuracy: {stats['accuracy']}%", use_container_width=True)
    else:
        st.warning("confusion_matrix.png not found")

    st.markdown('<div class="section-title">Model Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",     f"{stats['accuracy']}%")
    c2.metric("Samples",      f"{stats['total_samples']:,}")
    c3.metric("Attack Types", f"{stats['attack_types']}")
    c4.metric("Features",     f"{stats['features']}")

# ══════════════════════════════════
# TAB 5 — Project Overview
# ══════════════════════════════════
with tab5:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📌 Problem Statement")
        st.markdown("""
**Distributed Denial of Service (DDoS)** attacks are one of the most critical
threats to cloud infrastructure. Attackers flood servers with massive traffic
from multiple compromised sources — making services unavailable to real users.

**This system solves:**
- Detecting DDoS attacks in real-time from network traffic
- Classifying the exact type of attack (11 categories)
- Providing confidence scores for each prediction
- Visual analysis of attack patterns and model performance
        """)
        st.markdown("### 🔬 Methodology")
        st.markdown("""
1. **Data Collection** — CIC-DDoS2019 (Canadian Institute for Cybersecurity)
2. **Preprocessing** — Null removal, infinity handling, StandardScaler
3. **Feature Extraction** — 82 network flow parameters selected
4. **Model Training** — Random Forest (100 estimators, max_depth=20)
5. **Evaluation** — Confusion matrix, Precision, Recall, F1-score
6. **Deployment** — Streamlit Cloud — accessible anywhere via browser
        """)

    with col2:
        st.markdown("### 📊 Results Summary")
        st.markdown(f"""
| Metric | Model A | Model B |
|---|---|---|
| Accuracy | **92.59%** | **83.96%** |
| Samples | **53,549** | **87,536** |
| Attack Types | **11 + BENIGN** | **12 + BENIGN** |
| Features | **82** | **82** |
| Algorithm | **Random Forest** | **Random Forest** |
| Data Used | **01-12 only** | **01-12 + 03-11** |
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
| Model Storage | Joblib (.pkl files) |
| Version Control | GitHub |
        """)

# FOOTER
st.divider()
st.markdown(
    "<center style='color:#4a5568;font-size:0.8rem'>"
    "Distributed Denial of Service (DDoS) Protection System for Cloud"
    " &nbsp;|&nbsp; Problem Statement: SIH1649"
    " &nbsp;|&nbsp; CIC-DDoS2019 Dataset"
    "</center>", unsafe_allow_html=True)
