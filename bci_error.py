

# ==========================
# Brain-Computer Interface (BCI) Error Detection - Single Cell Pipeline
# ==========================

import os, zipfile
import numpy as np
import pandas as pd
from scipy.signal import iirnotch, lfilter, welch
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

# ------------------------------------------------
# 1. Load and Extract Training Data
# ------------------------------------------------
zip_path = "/kaggle/input/inria-bci-challenge/train.zip"
extract_dir = "/kaggle/working/train_data"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

csv_files = [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.csv')]
print(f"Found {len(csv_files)} training files.")

# Merge into single DataFrame
full_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
print("Training shape:", full_df.shape)

# ------------------------------------------------
# 2. Preprocessing (Notch Filter for Powerline Noise)
# ------------------------------------------------
def notch_filter(signal, fs=256, f0=50, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return lfilter(b, a, signal)

eeg_cols = [col for col in full_df.columns if col not in ['Time', 'EOG', 'FeedBackEvent']]
X_raw = full_df[eeg_cols].apply(lambda col: notch_filter(col))
y = full_df['FeedBackEvent'].values

# ------------------------------------------------
# 3. Artifact Removal (ICA + EOG Regression)
# ------------------------------------------------
# ICA decomposition
ica = FastICA(n_components=X_raw.shape[1], random_state=42)
S_ = ica.fit_transform(X_raw.values)
A_ = ica.mixing_

# Remove EOG-related components
eog_signal = full_df['EOG'].values
corrs = [np.corrcoef(S_[:, i], eog_signal)[0, 1] for i in range(S_.shape[1])]
S_clean = S_.copy()
for i, c in enumerate(corrs):
    if abs(c) > 0.3:
        S_clean[:, i] = 0
X_clean = np.dot(S_clean, A_.T)

# Regress out residual EOG
cleaned_df = full_df.copy()
for idx, ch in enumerate(eeg_cols):
    y_signal = X_clean[:, idx]
    model = LinearRegression().fit(eog_signal.reshape(-1, 1), y_signal)
    cleaned_df[ch] = y_signal - model.predict(eog_signal.reshape(-1, 1))

print("Artifact removal complete.")

# ------------------------------------------------
# 4. Feature Extraction
# ------------------------------------------------
X = cleaned_df[eeg_cols].values
X_features = pd.DataFrame({
    "var": X.var(axis=1),
    "mean": X.mean(axis=1),
    "std": X.std(axis=1)
})

print("Feature matrix shape:", X_features.shape)

# ------------------------------------------------
# 5. Train/Validation Split and Model Training
# ------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(X_features, y, test_size=0.2, random_state=42)

model = SVC(probability=True, random_state=42)
model.fit(X_train, y_train)

y_pred_val = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred_val)
print("Validation AUC:", auc)

# ------------------------------------------------
# 6. Load and Predict on Test Data
# ------------------------------------------------
test_zip_path = "/kaggle/input/inria-bci-challenge/test.zip"
all_dfs = []

with zipfile.ZipFile(test_zip_path, "r") as z:
    for filename in z.namelist():
        if filename.endswith(".csv"):
            with z.open(filename) as f:
                df = pd.read_csv(f)
                df["source_file"] = filename
                all_dfs.append(df)

test_df = pd.concat(all_dfs, ignore_index=True)
print("Test shape:", test_df.shape)

# Use same EEG cols as before (if exist)
test_eeg_cols = [c for c in test_df.columns if c in eeg_cols]
test_X = test_df[test_eeg_cols].values
test_features = pd.DataFrame({
    "var": test_X.var(axis=1),
    "mean": test_X.mean(axis=1),
    "std": test_X.std(axis=1)
})

# Predict
test_preds = model.predict(test_features)
print("Test predictions sample:", test_preds[:10])
