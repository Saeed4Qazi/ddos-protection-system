"""
SIH1649 - DDoS Protection System
Model A: Only 01-12 folder
Expected Accuracy: ~92.59%
Run: python3 save_model_9259.py
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
print("   Model A — Only 01-12 folder")
print("   Expected accuracy: ~92.59%")
print("=" * 55)

all_dfs = []
DATA_FOLDER = "./01-12"

print(f"\n   Loading from {DATA_FOLDER} only...")
for file in sorted(os.listdir(DATA_FOLDER)):
    if file.endswith(".csv"):
        filepath = os.path.join(DATA_FOLDER, file)
        try:
            df = pd.read_csv(filepath, nrows=5000, low_memory=False)
            df.columns = df.columns.str.strip()
            if 'Label' not in df.columns:
                df['Label'] = file.replace(".csv","")
            df['Label'] = df['Label'].str.strip()
            all_dfs.append(df)
            print(f"   OK {file} ({len(df)} rows)")
        except Exception as e:
            print(f"   SKIP {file}: {e}")

print("\n[2] Combining and cleaning...")
data = pd.concat(all_dfs, ignore_index=True)
data.columns = data.columns.str.strip()
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
print(f"   Total rows: {len(data):,}")
print(f"   Attack types: {data['Label'].nunique()}")
print(data['Label'].value_counts().to_string())

drop_cols = ['Label','Flow ID','Source IP','Destination IP','Timestamp','SimillarHTTP']
X = data.drop(columns=[c for c in drop_cols if c in data.columns], errors='ignore')
X = X.select_dtypes(include=[np.number])
le = LabelEncoder()
y  = le.fit_transform(data['Label'])

print(f"\n[3] Splitting 80/20... Features: {X.shape[1]}")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("\n[4] Training Random Forest (1-2 min)...")
model = RandomForestClassifier(
    n_estimators=100, max_depth=20,
    n_jobs=-1, random_state=42,
    class_weight='balanced')
model.fit(X_train_sc, y_train)
acc = accuracy_score(y_test, model.predict(X_test_sc))
print(f"   Accuracy: {acc*100:.2f}%")

print("\n[5] Saving to model_A/ folder...")
os.makedirs("model_A", exist_ok=True)
joblib.dump(model,              "model_A/ddos_model.pkl")
joblib.dump(scaler,             "model_A/scaler.pkl")
joblib.dump(le,                 "model_A/label_encoder.pkl")
joblib.dump(X.columns.tolist(), "model_A/feature_names.pkl")

stats = {
    "accuracy":      round(acc*100, 2),
    "total_samples": len(data),
    "attack_types":  len(le.classes_),
    "features":      len(X.columns),
    "classes":       list(le.classes_),
    "folders_used":  ["01-12"]
}
with open("model_A/stats.json", "w") as f:
    json.dump(stats, f)

print("\n" + "=" * 55)
print(f"   MODEL A ACCURACY: {acc*100:.2f}%")
print(f"   TOTAL SAMPLES:    {len(data):,}")
print(f"   SAVED TO:         model_A/")
print("=" * 55)
