# Distributed Denial of Service (DDoS) Protection System for Cloud

**B.Tech Final Year Project | Computer Science & Engineering**
**Problem Statement: SIH1649 | Smart India Hackathon 2025**

---

## Project Overview

This project implements a real-time **DDoS attack detection and classification system** for cloud infrastructure using Machine Learning. The system analyses network traffic flow features and classifies them into 11 distinct DDoS attack types using a Random Forest classifier trained on the CIC-DDoS2019 benchmark dataset.

---

## Features

- Real-time network traffic classification
- Detects 11 types of DDoS attacks + normal (BENIGN) traffic
- Interactive web dashboard with live prediction
- Two trained models for comparative analysis
- Visual charts — attack distribution, feature importance, confusion matrix

---

## Dataset

**CIC-DDoS2019** — Canadian Institute for Cybersecurity
- Folder `01-12` — December 1, 2018 traffic data
- Folder `03-11` — November 3, 2018 traffic data
- Total samples used: 87,536
- Features extracted: 82 network flow parameters

---

## Attack Types Covered

| Attack Type | Description |
|---|---|
| DrDoS_DNS | DNS Amplification Attack |
| DrDoS_LDAP | LDAP Amplification Attack |
| DrDoS_MSSQL | MSSQL Amplification Attack |
| DrDoS_NTP | NTP Amplification Attack |
| DrDoS_NetBIOS | NetBIOS Amplification Attack |
| DrDoS_SNMP | SNMP Amplification Attack |
| DrDoS_SSDP | SSDP Amplification Attack |
| DrDoS_UDP | UDP Flood Attack |
| Syn | SYN Flood Attack |
| TFTP | TFTP Amplification Attack |
| UDP-lag | UDP Lag Attack |
| BENIGN | Normal Traffic |

---

## Project Structure

```
project/
├── 01-12/                  # CIC-DDoS2019 CSV files (Dec 1)
├── 03-11/                  # CIC-DDoS2019 CSV files (Nov 3)
├── PCAP-01-12_0-0249/      # Raw PCAP packet files
├── model/                  # Model B — trained on both folders (83.96%)
│   ├── ddos_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── feature_names.pkl
│   └── stats.json
├── model_A/                # Model A — trained on 01-12 only (92.59%)
│   ├── ddos_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── feature_names.pkl
│   └── stats.json
├── dashboard.py            # Streamlit web dashboard
├── save_model.py           # Train Model B (both folders)
├── save_model_9259.py      # Train Model A (01-12 only)
├── ddos_model.py           # Standalone ML script
└── README.md               # This file
```

---

## Results

| Model | Dataset Used | Samples | Accuracy |
|---|---|---|---|
| Model A | 01-12 only | 53,549 | **92.59%** |
| Model B | 01-12 + 03-11 | 87,536 | **83.96%** |

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| ML Algorithm | Random Forest Classifier |
| Data Processing | Pandas, NumPy |
| ML Library | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Model Storage | Joblib |

---

## How to Run

### Step 1 — Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

### Step 2 — Train and save model (run only once)
```bash
# Model A — 01-12 only (92.59% accuracy)
python3 save_model_9259.py

# Model B — both folders (83.96% accuracy)
python3 save_model.py
```

### Step 3 — Launch dashboard
```bash
streamlit run dashboard.py
```

### Step 4 — Open in browser
```
http://localhost:8501
```

---

## Algorithm — Random Forest Classifier

- **Estimators:** 100 decision trees
- **Max Depth:** 20
- **Train/Test Split:** 80% / 20%
- **Feature Scaling:** StandardScaler
- **Class Weight:** Balanced (handles class imbalance)

---

## Developer

**Name:** Saeed Qazi
**Degree:** B.Tech — Computer Science & Engineering
**Project:** DDoS Protection System for Cloud (SIH1649)

---

*Dataset credit: Canadian Institute for Cybersecurity (CIC), University of New Brunswick*
