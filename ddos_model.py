"""
SIH1649 - DDoS Protection System
ML Model: Multi-class DDoS Attack Detection
Dataset: CIC-DDoS2019 (01-12 folder)
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns

# ──────────────────────────────────────────────
# STEP 1: LOAD ALL CSV FILES
# ──────────────────────────────────────────────

# !! CHANGE THIS PATH to your actual folder path !!
DATA_FOLDER = "./01-12"

print("=" * 55)
print("   SIH1649 - DDoS Detection System")
print("=" * 55)
print(f"\n[1] Loading CSV files from: {DATA_FOLDER}\n")

all_dfs = []
for file in os.listdir(DATA_FOLDER):
    if file.endswith(".csv"):
        filepath = os.path.join(DATA_FOLDER, file)
        try:
            # Read only 5000 rows per file to keep it fast
            df = pd.read_csv(filepath, nrows=5000, low_memory=False)
            # Add label from filename if no Label column
            if 'Label' not in df.columns and ' Label' not in df.columns:
                label = file.replace(".csv", "")
                df['Label'] = label
            all_dfs.append(df)
            print(f"   ✓ Loaded: {file}  ({len(df)} rows)")
        except Exception as e:
            print(f"   ✗ Skipped: {file}  ({e})")

print(f"\n   Total files loaded: {len(all_dfs)}")

# ──────────────────────────────────────────────
# STEP 2: COMBINE & CLEAN DATA
# ──────────────────────────────────────────────
print("\n[2] Combining and cleaning data...")

data = pd.concat(all_dfs, ignore_index=True)
print(f"   Total rows: {data.shape[0]}")
print(f"   Total columns: {data.shape[1]}")

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Show label distribution
print("\n   Attack type distribution:")
label_col = 'Label'
print(data[label_col].value_counts().to_string())

# Drop rows with NaN or Inf
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
print(f"\n   Rows after cleaning: {data.shape[0]}")

# ──────────────────────────────────────────────
# STEP 3: FEATURE SELECTION
# ──────────────────────────────────────────────
print("\n[3] Preparing features...")

# Drop non-numeric and identifier columns
drop_cols = ['Label', 'Flow ID', 'Source IP', 'Destination IP',
             'Timestamp', 'SimillarHTTP']

# Keep only numeric columns
X = data.drop(columns=[c for c in drop_cols if c in data.columns],
              errors='ignore')
X = X.select_dtypes(include=[np.number])

# Encode labels
le = LabelEncoder()
y = le.fit_transform(data[label_col])

print(f"   Features used: {X.shape[1]}")
print(f"   Classes: {list(le.classes_)}")

# ──────────────────────────────────────────────
# STEP 4: TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
print("\n[4] Splitting data (80% train, 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train size: {X_train.shape[0]}")
print(f"   Test size:  {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ──────────────────────────────────────────────
# STEP 5: TRAIN RANDOM FOREST MODEL
# ──────────────────────────────────────────────
print("\n[5] Training Random Forest model...")
print("   (This may take 1-2 minutes...)\n")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    n_jobs=-1,          # Use all CPU cores
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train_sc, y_train)
print("   ✓ Model trained!")

# ──────────────────────────────────────────────
# STEP 6: EVALUATE
# ──────────────────────────────────────────────
print("\n[6] Evaluating model...\n")

y_pred = model.predict(X_test_sc)
acc = accuracy_score(y_test, y_pred)

print(f"   ACCURACY: {acc * 100:.2f}%\n")
print("   Classification Report:")
print("   " + "-" * 50)
report = classification_report(y_test, y_pred,
                                target_names=le.classes_)
for line in report.split('\n'):
    print("   " + line)

# ──────────────────────────────────────────────
# STEP 7: SAVE PLOTS
# ──────────────────────────────────────────────
print("\n[7] Saving charts...\n")

# -- Chart 1: Confusion Matrix --
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title('Confusion Matrix — DDoS Attack Detection', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("   ✓ Saved: confusion_matrix.png")

# -- Chart 2: Feature Importance --
feat_imp = pd.Series(model.feature_importances_,
                     index=X.columns).sort_values(ascending=False)
top15 = feat_imp.head(15)

plt.figure(figsize=(10, 6))
top15.plot(kind='barh', color='steelblue')
plt.gca().invert_yaxis()
plt.title('Top 15 Most Important Features', fontsize=14)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.close()
print("   ✓ Saved: feature_importance.png")

# -- Chart 3: Attack Distribution --
plt.figure(figsize=(10, 5))
data[label_col].value_counts().plot(kind='bar', color='coral',
                                     edgecolor='black')
plt.title('DDoS Attack Type Distribution', fontsize=14)
plt.xlabel('Attack Type')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('attack_distribution.png', dpi=150)
plt.close()
print("   ✓ Saved: attack_distribution.png")

# ──────────────────────────────────────────────
# STEP 8: DEMO — Predict single sample
# ──────────────────────────────────────────────
print("\n[8] Demo prediction on a random sample...")

sample = X_test_sc[0].reshape(1, -1)
pred_class = le.inverse_transform(model.predict(sample))[0]
actual_class = le.inverse_transform([y_test[0]])[0]

print(f"\n   Actual attack type  : {actual_class}")
print(f"   Predicted by model  : {pred_class}")
print(f"   Correct prediction  : {'✓ YES' if pred_class == actual_class else '✗ NO'}")

# ──────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"   FINAL ACCURACY: {acc * 100:.2f}%")
print("   All charts saved in project folder!")
print("=" * 55)
print("\nDone! Ab SIH mein yeh charts aur accuracy present karo!")
