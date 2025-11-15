# client_utils.py
"""
FDDT utilities: load CSV, detrend (local drift removal), build sequence windows,
and helpers to compute thresholds from reconstruction errors.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# PARAMETERS (tweakable)
SEQ_LEN = 50          # number of consecutive rows per sample
ROLL_MEDIAN = 251     # window for long-run median (drift estimation). If file shorter, min_periods used.
TEST_SPLIT = 0.2

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # Remove unnamed and timestamp-like cols
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    for c in ("timestamp", "time", "date"):
        if c in df.columns:
            df = df.drop(columns=[c])
    if "machine_status" not in df.columns:
        raise ValueError("CSV must contain 'machine_status' column")
    # Standardize label
    df["machine_status"] = df["machine_status"].astype(str).str.upper().map(lambda s: 0 if s == "NORMAL" else 1)
    return df

def detrend_by_long_median(X: pd.DataFrame, window:int = ROLL_MEDIAN):
    """
    Estimate a long-run baseline per sensor using rolling median,
    then compute residuals (X - baseline). Returns residual array and baseline (both as numpy).
    Using median helps robustly capture slow drift.
    """
    baseline = X.rolling(window=window, min_periods=1, center=False).median()
    residual = X - baseline
    # Fill any remaining NaNs
    residual = residual.fillna(method="bfill").fillna(method="ffill").fillna(0.0)
    baseline = baseline.fillna(method="bfill").fillna(method="ffill").fillna(0.0)
    return residual.values.astype(np.float32), baseline.values.astype(np.float32)

def make_sequences(X: np.ndarray, seq_len: int = SEQ_LEN):
    """Create sliding sequences of shape (N, seq_len, features)."""
    if len(X) < seq_len + 1:
        # Not enough rows: pad by repeating last row to create a single sequence
        if len(X) == 0:
            raise ValueError("No numeric sensor data available")
        pad = np.tile(X[-1], (seq_len, 1))
        return pad.reshape(1, seq_len, X.shape[1])
    seqs = []
    for i in range(0, len(X) - seq_len):
        seqs.append(X[i : i + seq_len])
    return np.array(seqs, dtype=np.float32)

def load_and_prepare_sequences(cid:int, csv_path: str = None, seq_len:int = SEQ_LEN):
    if csv_path is None:
        csv_path = f"client_{cid}.csv"
        if not os.path.exists(csv_path):
            # Try to find a .txt file
            txt_path = f"client_{cid}.txt"
            if os.path.exists(txt_path):
                # Convert txt → csv
                df = pd.read_csv(txt_path, sep=None, engine='python')  # auto-detect separator
                df.to_csv(csv_path, index=False)
            else:
                raise FileNotFoundError(f"{csv_path} not found.")
    df = pd.read_csv(csv_path)
    df = _clean_df(df)

    # sensors are all columns except machine_status
    sensor_cols = [c for c in df.columns if c != "machine_status"]
    if len(sensor_cols) == 0:
        raise ValueError("No sensor columns found in CSV")
    sensors = df[sensor_cols].apply(pd.to_numeric, errors="coerce")
    sensors = sensors.fillna(sensors.median())

    residuals, baseline = detrend_by_long_median(sensors)

    labels = df["machine_status"].astype(int).values

    # Build sequences from residuals; we will associate a sequence's label as the label at the sequence end
    X_seq = make_sequences(residuals, seq_len=seq_len)
    y_seq = []
    for i in range(len(residuals) - seq_len):
        y_seq.append(labels[i + seq_len])
    y_seq = np.array(y_seq, dtype=np.int64)

    # For autoencoder training, use only sequences where label == 0 (NORMAL)
    mask_normal = (y_seq == 0)
    X_normal = X_seq[mask_normal]
    # If none are normal, fallback: use all sequences
    if len(X_normal) == 0:
        X_normal = X_seq

    # Train/test split by time ordering (use last portion as test to simulate future)
    split_idx = int(len(X_normal) * 0.8) if len(X_normal) > 2 else 1
    x_train = X_normal[:split_idx]
    x_hold = X_normal[split_idx:] if split_idx < len(X_normal) else X_normal[split_idx:split_idx+1]

    # For evaluation after federated aggregation, we keep a global test set composed of recent sequences (both normal and abnormal)
    # Here: take the last 20% of all sequences
    split_all = int(len(X_seq) * (1.0 - TEST_SPLIT))
    x_test = X_seq[split_all:] if split_all < len(X_seq) else X_seq[-1:].copy()
    y_test = y_seq[split_all:] if split_all < len(y_seq) else y_seq[-1:].copy()

    # Standard scale residuals per feature (fit on training residuals flattened)
    # Flatten train sequences to shape (-1, features) for scaler fit
    scaler = StandardScaler()
    flat_train = x_train.reshape((-1, x_train.shape[-1]))
    scaler.fit(flat_train)
    # transform train, hold, test
    def apply_scaler(arr):
        if arr.size == 0:
            return arr
        s = arr.reshape((-1, arr.shape[-1]))
        s = scaler.transform(s)
        return s.reshape(arr.shape)
    x_train = apply_scaler(x_train)
    x_hold = apply_scaler(x_hold)
    x_test = apply_scaler(x_test)

    input_shape = (seq_len, x_train.shape[-1]) if x_train.size>0 else (seq_len, sensors.shape[1])
    # return also a small metadata dict
    meta = {
        "sensor_cols": sensor_cols,
        "n_train_seq": len(x_train),
        "n_hold_seq": len(x_hold),
        "n_test_seq": len(x_test)
    }
    return x_train, x_hold, x_test, y_test, input_shape, scaler, meta
