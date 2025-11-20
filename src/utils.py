# src/utils.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os

def load_data(path="data/simulated_timeseries.csv"):
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df

def make_scalers(df, feature_cols, target_col, scaler_dir="models"):
    os.makedirs(scaler_dir, exist_ok=True)
    X_scaler = StandardScaler().fit(df[feature_cols].values)
    y_scaler = StandardScaler().fit(df[[target_col]].values)
    joblib.dump(X_scaler, os.path.join(scaler_dir, "X_scaler.joblib"))
    joblib.dump(y_scaler, os.path.join(scaler_dir, "y_scaler.joblib"))
    return X_scaler, y_scaler

def window_data(df, feature_cols, target_col, window_size=48, horizon=1, step=1, flatten=False):
    """
    Create sliding windows X and targets y.
    X shape: (n_samples, window_size, n_features)
    y shape: (n_samples, 1)
    horizon: number of steps ahead to predict (default 1)
    step: step between windows
    """
    arr_X = df[feature_cols].values
    arr_y = df[target_col].values.reshape(-1,1)
    Xs, ys, times = [], [], []
    for start in range(0, len(df) - window_size - horizon + 1, step):
        end = start + window_size
        Xs.append(arr_X[start:end])
        ys.append(arr_y[end + horizon - 1])
        times.append(df["time"].iloc[end + horizon - 1])
    Xs = np.array(Xs)
    ys = np.array(ys).reshape(-1,1)
    return Xs, ys, np.array(times)

def scale_windows(X, y, X_scaler, y_scaler):
    # X: (n, window, features) -> flatten for scaler then reshape back
    n, w, f = X.shape
    X_flat = X.reshape(n, w*f)
    X_scaled_flat = X_scaler.transform(X_flat)
    X_scaled = X_scaled_flat.reshape(n, w, f)
    y_scaled = y_scaler.transform(y)
    return X_scaled, y_scaled

