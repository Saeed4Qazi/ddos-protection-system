"""
SIH1649 - DDoS Protection System
Train model with BOTH 01-12 and 03-11 folders
Labels standardized for consistent accuracy
Run: python3 save_model.py
"""

import pandas as pd
import numpy as np
import os
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

print("=" * 55)
print("   DDoS Protection System - Model Trainer")
print("   Using: 01-12 + 03-11 folders")
print("=" * 55)

# Label mapping — fix 03-11 label names to match 01-12
LABEL_MAP = {
    "LDAP":    "DrDoS_LDAP",
    "MSSQL":   "DrDoS_MSSQL",
    "NetBIOS": "DrDoS_NetBIOS",
    "Portmap": "DrDoS_UDP",
    "UDP":     "DrDoS_UDP",
    "UDPLag":  "UDP-lag",
    "Syn":     "Syn",
    "BENIGN":  "BENIGN",
}

all_dfs = []

for folder in ["./01-12", "./03-11"]:
    if not os.path.exists(folder):
        print(f"\n   Skipping {folder} (not found)")
        continue
    print(f"\n   Loading from {folder}...")
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            filepath = os.path.join(folder, file)
            try:
                df = pd.read_csv(filepath, nrows=5000, low_memory=False)
                df.columns = df.columns.str.strip()

                # Get label
                if 'Label' in df.columns:
                    label_col = 'Label'
                elif ' Label' in df.columns:
                    label_col = ' Label'
                    df.rename(columns={' Label': 'Label'}, inplace=True)
                    label_col = 'Label'
                else:
                    raw = file.replace(".csv","")
                    df['Label'] = raw
                    label_col = 'Label'

                # Standardize labels from 03-11
                df['Label'] = df['Label'].str.strip()
                df['Label'] = df['Label'].replace(LABEL_MAP)

                all_dfs.append(df)
                unique_label = df['Label'].iloc[0]
                print(f"   OK {file} ({len(df)} rows) → [{unique_label}]")
            except Exception as e:
                print(f"   SKIP {file}: {e}")

print("\n[2] Combining and cleaning...")
data = pd.concat(all_dfs, ignore_index=True)
data.columns = data.columns.str.strip()
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
print(f"   Total rows: {len(data):,}")
print(f"\n   Attack type distribution:")
print(data['Label'].value_counts().to_string())

drop_cols = ['Label','Flow ID','Source IP','Destination IP',
             'Timestamp','SimillarHTTP']
X = data.drop(columns=[c for c in drop_cols if c in data.columns], errors='ignore')
X = X.select_dtypes(include=[np.number])
le = LabelEncoder()
y  = le.fit_transform(data['Label'])

print(f"\n[3] Splitting 80/20...")
print(f"   Features: {X.shape[1]}")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("\n[4] Training Random Forest (2-3 min)...")
model = RandomForestClassifier(
    n_estimators=100, max_depth=20,
    n_jobs=-1, random_state=42,
    class_weight='balanced')
model.fit(X_train_sc, y_train)
acc = accuracy_score(y_test, model.predict(X_test_sc))
print(f"   Model trained! Accuracy: {acc*100:.2f}%")

print("\n[5] Saving model files...")
os.makedirs("model", exist_ok=True)
joblib.dump(model,              "model/ddos_model.pkl")
joblib.dump(scaler,             "model/scaler.pkl")
joblib.dump(le,                 "model/label_encoder.pkl")
joblib.dump(X.columns.tolist(), "model/feature_names.pkl")

stats = {
    "accuracy":      round(acc*100, 2),
    "total_samples": len(data),
    "attack_types":  len(le.classes_),
    "features":      len(X.columns),
    "classes":       list(le.classes_)
}
with open("model/stats.json", "w") as f:
    json.dump(stats, f)

print("   Saved: model/ddos_model.pkl")
print("   Saved: model/scaler.pkl")
print("   Saved: model/label_encoder.pkl")
print("   Saved: model/stats.json")
print("\n" + "=" * 55)
print(f"   FINAL ACCURACY: {acc*100:.2f}%")
print(f"   TOTAL SAMPLES:  {len(data):,}")
print("   Now run: streamlit run dashboard.py")
print("=" * 55)
