"""Aggregate per-period CSVs into waveform arrays (X.npy, y.npy).

Reads the period CSVs produced by data_process.py from data/{H2S,CH4,CO2,AIR}/,
extracts the detrended-resistance column, pads all waveforms to equal length,
and saves the full array to data/converted/. The shipped data/preprocessed/ is left
untouched; point train scripts at data/converted/ to use the from-scratch array.
"""
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
SAVE_DIR = "data/converted"
os.makedirs(SAVE_DIR, exist_ok=True)

label_map = {"H2S": 0, "CH4": 1, "CO2": 2, "AIR": 3}


def load_data():
    X, y = [], []
    for label_name, label_id in label_map.items():
        class_dir = os.path.join(DATA_DIR, label_name)
        if not os.path.isdir(class_dir):
            print(f"[skip] class folder not found: {label_name}")
            continue

        for fname in tqdm(sorted(os.listdir(class_dir)), desc=f"loading {label_name}"):
            if not fname.endswith(".csv"):
                continue

            fpath = os.path.join(class_dir, fname)
            try:
                df = pd.read_csv(fpath)
                r = df["R(OHM)_detrended"].values
                if len(r) < 50 or np.any(np.isnan(r)):
                    continue
                X.append(r)
                y.append(label_id)
            except Exception as e:
                print(f"[error] {fpath}: {e}")
    return X, y


def pad_sequences(X):
    max_len = max(len(seq) for seq in X)
    X_pad = np.zeros((len(X), max_len), dtype=np.float32)
    for i, seq in enumerate(X):
        seq = np.asarray(seq, dtype=np.float32)
        if len(seq) < max_len:
            X_pad[i, :len(seq)] = seq
        else:
            X_pad[i] = seq[:max_len]
    return X_pad


# 1. load per-period waveforms
X, y = load_data()
y = np.array(y)
print(f"total samples: {len(y)}")

# 2. pad to equal length
X_pad = pad_sequences(X)
print(f"padded shape: {X_pad.shape}")

# 3. 9:1 stratified split (kept for parity with the original pipeline)
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y, test_size=0.1, random_state=42, stratify=y
)

# 4. save
np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train)
np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)
np.save(os.path.join(SAVE_DIR, "X_test.npy"), X_test)
np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)
np.save(os.path.join(SAVE_DIR, "X.npy"), X_pad)
np.save(os.path.join(SAVE_DIR, "y.npy"), y)

print(f"done: X_train={X_train.shape}, X_test={X_test.shape}, X={X_pad.shape}")
