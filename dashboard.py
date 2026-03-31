"""
Distributed Denial of Service (DDoS) Protection System for Cloud
Final Year B.Tech Project | SIH1649
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="DDoS Protection System | B.Tech Final Year Project",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────
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
    .main-header h1 {
        color: #00d4ff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: 0.5px;
    }
    .main-header .subtitle {
        color: #ccd6f6;
        font-size: 0.95rem;
        margin: 4px 0;
    }
    .main-header .meta {
        color: #8892b0;
        font-size: 0.82rem;
        margin: 10px 0 0 0;
        letter-spacing: 0.5px;
    }

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
        color: #00d4ff !important; font-size: 1.8rem !important;
        font-weight: 800 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ Distributed Denial of Service (DDoS) Protection System for Cloud</h1>
    <p class="subtitle">
        Real-time Multi-class Attack Detection and Classification using Machine Learning
    </p>
    <p class="meta">
        B.Tech Final Year Project &nbsp;|&nbsp; Computer Science &amp; Engineering
        &nbsp;|&nbsp; Problem Statement: SIH1649
        &nbsp;|&nbsp; Dataset: CIC-DDoS2019 (Canadian Institute for Cybersecurity)
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ System Info")
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
    A **Distributed Denial of Service** attack floods a server
    with traffic from multiple sources, making it unavailable
    to legitimate users.
    """)

# ── Load & train ──────────────────────────────────────────
@st.cache_resource
def load_and_train():
    DATA_FOLDER = "./01-12"
    all_dfs = []
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".csv"):
            filepath = os.path.join(DATA_FOLDER, file)
            try:
                df = pd.read_csv(filepath, nrows=5000, low_memory=False)
                if 'Label' not in df.columns and ' Label' not in df.columns:
                    df['Label'] = file.replace(".csv","")
                all_dfs.append(df)
            except:
                pass
    data = pd.concat(all_dfs, ignore_index=True)
    data.columns = data.columns.str.strip()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    drop_cols = ['Label','Flow ID','Source IP','Destination IP','Timestamp','SimillarHTTP']
    X = data.drop(columns=[c for c in drop_cols if c in data.columns], errors='ignore')
    X = X.select_dtypes(include=[np.number])
    le = LabelEncoder()
    y  = le.fit_transform(data['Label'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=20, n_jobs=-1,
        random_state=42, class_weight='balanced')
    model.fit(X_train_sc, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_sc))
    return model, scaler, le, X, data, acc, X_test_sc, y_test

with st.spinner("🔄 Loading CIC-DDoS2019 dataset and training model..."):
    model, scaler, le, X, data, acc, X_test_sc, y_test = load_and_train()

# ── Metrics ───────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("📦 Total Samples",       f"{len(data):,}")
c2.metric("⚠️ DDoS Attack Types",   f"{len(le.classes_)}")
c3.metric("🎯 Detection Accuracy",  f"{acc*100:.2f}%")
c4.metric("🔬 Network Features",    f"{X.shape[1]}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Live Detection",
    "📊 Data Analysis",
    "🎯 Model Performance",
    "📖 Project Overview"
])

# ═══════════════════════════
# TAB 1 — Live Detection
# ═══════════════════════════
with tab1:
    st.markdown('<div class="section-title">Real-time DDoS attack classification</div>',
                unsafe_allow_html=True)
    st.caption("Adjust the network traffic parameters below and click Predict to classify the traffic.")

    feature_names = X.columns.tolist()
    top_features  = pd.Series(model.feature_importances_, index=feature_names)\
                      .sort_values(ascending=False).head(8).index.tolist()

    inputs = {}
    cols = st.columns(4)
    for i, feat in enumerate(top_features):
        mn = float(X[feat].quantile(0.05))
        mx = float(X[feat].quantile(0.95))
        dv = float(X[feat].median())
        if mn == mx: mx = mn + 1
        inputs[feat] = cols[i % 4].slider(feat[:22], mn, mx, dv, key=f"s_{feat}")

    st.markdown("")
    left_p, right_p = st.columns([1, 1])

    with left_p:
        if st.button("🔍  Classify Traffic", use_container_width=True, type="primary"):
            row = pd.DataFrame([{f: X[f].median() for f in feature_names}])
            for feat, val in inputs.items():
                row[feat] = val
            row_sc = scaler.transform(row)
            pred   = model.predict(row_sc)[0]
            proba  = model.predict_proba(row_sc)[0]
            label  = le.inverse_transform([pred])[0]
            conf   = proba[pred] * 100

            st.markdown("")
            if label == "BENIGN":
                st.markdown(
                    f'<div class="alert-safe">✅ BENIGN — Normal Traffic Detected<br>'
                    f'<span style="font-size:0.88rem;font-weight:400">'
                    f'Confidence: {conf:.1f}% &nbsp;|&nbsp; No threat identified</span></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="alert-danger">🚨 DDoS ATTACK DETECTED: {label}<br>'
                    f'<span style="font-size:0.88rem;font-weight:400">'
                    f'Confidence: {conf:.1f}% &nbsp;|&nbsp; Immediate mitigation required</span></div>',
                    unsafe_allow_html=True)

            st.markdown("#### Classification Probabilities")
            top3 = np.argsort(proba)[::-1][:3]
            for idx in top3:
                name  = le.inverse_transform([idx])[0]
                pct   = proba[idx] * 100
                color = "#00ff88" if name == "BENIGN" else "#ff4757"
                bar   = int(pct / 4)
                st.markdown(
                    f"**{name}** &nbsp; `{pct:.1f}%` &nbsp;"
                    f"<span style='color:{color}'>{'█' * bar}</span>",
                    unsafe_allow_html=True)

    with right_p:
        st.markdown("#### 🔎 Feature Explanation")
        st.markdown("""
        | Parameter | Meaning |
        |---|---|
        | Flow Duration | How long the connection lasted |
        | Packet Length | Size of data packets |
        | Fwd Packets/s | Packets sent per second |
        | SYN Flag Count | Connection request count |
        | Bwd IAT Mean | Time between incoming packets |

        ---
        **Try this demo:**
        > Drag `SYN Flag Count` to maximum
        > → Click Classify
        > → Model detects `Syn` flood attack!
        """)

# ═══════════════════════════
# TAB 2 — Data Analysis
# ═══════════════════════════
with tab2:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">DDoS attack type distribution</div>',
                    unsafe_allow_html=True)
        counts = data['Label'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        fig1.patch.set_facecolor('#0e1117')
        ax1.set_facecolor('#1a1f35')
        colors = plt.cm.cool(np.linspace(0.2, 0.9, len(counts)))
        ax1.bar(counts.index, counts.values, color=colors,
                edgecolor='#00d4ff44', linewidth=0.8)
        ax1.set_xlabel("Attack Type", color='#8892b0')
        ax1.set_ylabel("Number of Samples", color='#8892b0')
        ax1.set_title("CIC-DDoS2019 — Attack Distribution", color='#ccd6f6', pad=12)
        ax1.tick_params(colors='#8892b0')
        for s in ['top','right']: ax1.spines[s].set_visible(False)
        for s in ['bottom','left']: ax1.spines[s].set_color('#1e2a3a')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig1)

    with col_b:
        st.markdown('<div class="section-title">Top 10 most significant features</div>',
                    unsafe_allow_html=True)
        feat_imp = pd.Series(model.feature_importances_,
                             index=X.columns).sort_values(ascending=False).head(10)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        fig2.patch.set_facecolor('#0e1117')
        ax2.set_facecolor('#1a1f35')
        feat_imp.plot(kind='barh', ax=ax2, color='#00d4ff', alpha=0.85)
        ax2.invert_yaxis()
        ax2.set_title("Random Forest — Feature Importance", color='#ccd6f6', pad=12)
        ax2.set_xlabel("Importance Score", color='#8892b0')
        ax2.tick_params(colors='#8892b0', labelsize=9)
        for s in ['top','right']: ax2.spines[s].set_visible(False)
        for s in ['bottom','left']: ax2.spines[s].set_color('#1e2a3a')
        plt.tight_layout()
        st.pyplot(fig2)

# ═══════════════════════════
# TAB 3 — Model Performance
# ═══════════════════════════
with tab3:
    st.markdown('<div class="section-title">Confusion matrix — predicted vs actual</div>',
                unsafe_allow_html=True)
    y_pred = model.predict(X_test_sc)
    cm     = confusion_matrix(y_test, y_pred)
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    fig3.patch.set_facecolor('#0e1117')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_,
                ax=ax3, linewidths=0.5, linecolor='#0e1117',
                annot_kws={"size": 9})
    ax3.set_xlabel("Predicted Label", color='#ccd6f6', fontsize=11)
    ax3.set_ylabel("Actual Label",    color='#ccd6f6', fontsize=11)
    ax3.set_title(
        f"Confusion Matrix  |  Overall Detection Accuracy: {acc*100:.2f}%",
        color='#00d4ff', fontsize=13, pad=14)
    ax3.tick_params(colors='#8892b0', labelsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig3)

    st.markdown('<div class="section-title">Per-class precision, recall and F1-score</div>',
                unsafe_allow_html=True)
    report = classification_report(y_test, y_pred,
                                   target_names=le.classes_, output_dict=True)
    df_rep = pd.DataFrame(report).transpose()
    df_rep = df_rep[df_rep.index.isin(list(le.classes_))]
    st.dataframe(
        df_rep[['precision','recall','f1-score','support']]
        .style.background_gradient(cmap='Blues',
                                   subset=['precision','recall','f1-score'])
        .format({'precision':'{:.2f}','recall':'{:.2f}',
                 'f1-score':'{:.2f}','support':'{:.0f}'}),
        use_container_width=True)

# ═══════════════════════════
# TAB 4 — Project Overview
# ═══════════════════════════
with tab4:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📌 Problem Statement")
        st.markdown("""
        **Distributed Denial of Service (DDoS)** attacks are one of the most
        critical threats to cloud infrastructure. Attackers flood servers with
        massive amounts of traffic from multiple sources simultaneously, causing
        legitimate users to be denied access.

        This project proposes an **ML-based real-time detection system** that:
        - Analyses live network traffic flow features
        - Classifies traffic as benign or one of 11 DDoS attack types
        - Provides instant alerts for detected attacks
        """)

        st.markdown("### 🔬 Methodology")
        st.markdown("""
        1. **Data Collection** — CIC-DDoS2019 dataset (Canadian Institute for Cybersecurity)
        2. **Preprocessing** — Null removal, infinity handling, feature scaling
        3. **Feature Extraction** — 82 network flow parameters (packet size, flags, rates)
        4. **Model Training** — Random Forest Classifier (100 estimators)
        5. **Evaluation** — Confusion matrix, precision, recall, F1-score
        6. **Deployment** — Real-time web dashboard using Streamlit
        """)

    with col2:
        st.markdown("### 📊 Results Summary")
        st.markdown(f"""
        | Metric | Value |
        |---|---|
        | Overall Accuracy | **{acc*100:.2f}%** |
        | Total Training Samples | **{int(len(data)*0.8):,}** |
        | Total Test Samples | **{int(len(data)*0.2):,}** |
        | Features Used | **82** |
        | Attack Types Classified | **11 + BENIGN** |
        | Algorithm | **Random Forest** |
        """)

        st.markdown("### ⚙️ Tech Stack")
        st.markdown("""
        | Layer | Technology |
        |---|---|
        | Language | Python 3.12 |
        | ML Library | Scikit-learn |
        | Data Processing | Pandas, NumPy |
        | Visualization | Matplotlib, Seaborn |
        | Dashboard | Streamlit |
        | Dataset | CIC-DDoS2019 |
        """)

st.divider()
st.markdown(
    "<center style='color:#4a5568;font-size:0.8rem'>"
    "Distributed Denial of Service (DDoS) Protection System for Cloud &nbsp;|&nbsp; "
    "B.Tech Final Year Project &nbsp;|&nbsp; "
    "Problem Statement SIH1649 &nbsp;|&nbsp; "
    "CIC-DDoS2019 Dataset"
    "</center>",
    unsafe_allow_html=True)
