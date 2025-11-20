# src/train.py
"""
Training pipeline with:
- data loading
- scalers creation
- windowing
- rolling-origin cross validation
- model training and evaluation
- Optuna hyperparameter tuning
- baseline comparison
- SHAP explainability (optional, can be slow)
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
import joblib
import optuna
from tqdm import tqdm

from utils import load_data, make_scalers, window_data, scale_windows
from models import build_probabilistic_lstm, predict_mu_sigma
from evaluate import rmse, mae, mape, crps_score
from baseline import fit_ets, fit_arima, forecast_model
from explainability import explain_shap

import matplotlib.pyplot as plt

# --- Config ---
DATA_PATH = "data/simulated_timeseries.csv"
FEATURE_COLS = ["feature1", "feature2", "spikes"]
TARGET_COL = "target"
WINDOW_SIZE = 48
HORIZON = 1
STEP = 1
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load data ---
df = load_data(DATA_PATH)
print("Data loaded:", df.shape)

# --- Build scalers ---
X_scaler, y_scaler = make_scalers(df, FEATURE_COLS, TARGET_COL, scaler_dir=MODEL_DIR)

# --- Windowing ---
X, y, times = window_data(df, FEATURE_COLS, TARGET_COL, window_size=WINDOW_SIZE, horizon=HORIZON, step=STEP)
print("Windows:", X.shape, y.shape)

# --- Scale windows ---
X_scaled, y_scaled = scale_windows(X, y, X_scaler, y_scaler)

# --- Rolling-origin CV configuration ---
# We'll create several folds with expanding training window
n_samples = X_scaled.shape[0]
initial_train = int(0.5 * n_samples)  # start with 50% for training
fold_step = int(0.1 * n_samples)      # each fold expands by 10% of data
folds = []
start = initial_train
while start + fold_step < n_samples:
    train_idx = np.arange(0, start)
    val_idx = np.arange(start, start + fold_step)
    folds.append((train_idx, val_idx))
    start += fold_step
print(f"Created {len(folds)} rolling folds.")

# --- Optuna objective ---
def objective(trial):
    units = trial.suggest_int("units", 32, 128)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    layers = trial.suggest_int("layers", 1, 3)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    batch = trial.suggest_categorical("batch", [16, 32, 64])

    metrics_fold = []
    for (train_idx, val_idx) in folds[:2]:  # for speed in tuning we use first 2 folds
        X_tr, y_tr = X_scaled[train_idx], y_scaled[train_idx]
        X_val, y_val = X_scaled[val_idx], y_scaled[val_idx]

        model = build_probabilistic_lstm(WINDOW_SIZE, len(FEATURE_COLS), units=units, dropout=dropout, lstm_layers=layers)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=None)  # loss set in model; recompile to set lr

        # fit
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=10, batch_size=batch, verbose=0)
        preds = model.predict(X_val)
        mu = preds[:,0:1]
        log_var = preds[:,1:2]
        sigma = np.sqrt(np.exp(log_var))
        # inverse-scale (y_scaler expects 2D)
        mu_inv = joblib.load(f"{MODEL_DIR}/y_scaler.joblib").inverse_transform(mu)
        y_val_inv = joblib.load(f"{MODEL_DIR}/y_scaler.joblib").inverse_transform(y_val)
        # compute RMSE on this fold
        fold_rmse = rmse(y_val_inv, mu_inv)
        metrics_fold.append(fold_rmse)
    return np.mean(metrics_fold)

# Run Optuna
import tensorflow as tf
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=12, show_progress_bar=True)
print("Best params:", study.best_params)

best_params = study.best_params

# --- Train final model on full training data (last fold's train + val) ---
# Use last fold's indices to define final training slice
last_train_end = folds[-1][1][-1] if len(folds) > 0 else int(0.8 * n_samples)
X_train_final = X_scaled[:last_train_end]
y_train_final = y_scaled[:last_train_end]
X_test_final = X_scaled[last_train_end:]
y_test_final = y_scaled[last_train_end:]
print("Final train/test shapes:", X_train_final.shape, X_test_final.shape)

final_model = build_probabilistic_lstm(WINDOW_SIZE, len(FEATURE_COLS),
                                       units=best_params['units'],
                                       dropout=best_params['dropout'],
                                       lstm_layers=best_params['layers'])
final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['lr']), loss=None)
final_model.fit(X_train_final, y_train_final, validation_data=(X_test_final, y_test_final), epochs=30, batch_size=best_params['batch'], verbose=1)

# --- Evaluate final model ---
preds_final = final_model.predict(X_test_final)
mu_pred = preds_final[:,0:1]
log_var_pred = preds_final[:,1:2]
sigma_pred = np.sqrt(np.exp(log_var_pred))

# invert scaling
y_scaler = joblib.load(f"{MODEL_DIR}/y_scaler.joblib")
mu_inv = y_scaler.inverse_transform(mu_pred)
sigma_inv = sigma_pred * y_scaler.scale_[0]  # Rough scaling for sigma (since we used StandardScaler)
y_test_inv = y_scaler.inverse_transform(y_test_final)

from evaluate import rmse, mae, mape, crps_score
rmse_val = rmse(y_test_inv, mu_inv)
mae_val = mae(y_test_inv, mu_inv)
mape_val = mape(y_test_inv, mu_inv)
crps_val = crps_score(y_test_inv, mu_inv, sigma_inv)
print(f"Final metrics: RMSE={rmse_val:.4f}, MAE={mae_val:.4f}, MAPE={mape_val:.2f}%, CRPS={crps_val:.4f}")

# --- Baseline comparison: ETS and ARIMA on the target series using last portion
target_series = df[TARGET_COL]
train_series = target_series.iloc[:last_train_end + WINDOW_SIZE]  # align with windows
test_series = target_series.iloc[last_train_end + WINDOW_SIZE:]

ets_fit = fit_ets(train_series)
ets_pred = forecast_model(ets_fit, len(test_series))
arima_fit = fit_arima(train_series)
arima_pred = forecast_model(arima_fit, len(test_series))

# Metrics for baselines
ets_rmse = rmse(test_series.values, ets_pred)
arima_rmse = rmse(test_series.values, arima_pred)
print("Baseline RMSEs: ETS:", ets_rmse, "ARIMA:", arima_rmse)

# CRPS for baseline: approximate by assuming normal with small constant sigma
baseline_sigma = np.std(train_series.values - ets_fit.fittedvalues[:len(train_series)]) + 1e-6
ets_crps = crps_score(test_series.values, ets_pred, np.ones_like(ets_pred)*baseline_sigma)
arima_crps = crps_score(test_series.values, arima_pred, np.ones_like(arima_pred)*baseline_sigma)
print("Baseline CRPS: ETS:", ets_crps, "ARIMA:", arima_crps)

# --- Save final model
final_model.save(os.path.join(MODEL_DIR, "final_probabilistic_lstm.h5"))

# --- SHAP explainability (optional)
try:
    # Take a small sample for explanation
    shap_path = explain_shap(final_model, X_train_final[:200], X_test_final[:100], feature_names=None, save_path="shap_summary.png")
    print("Saved SHAP summary at", shap_path)
except Exception as e:
    print("SHAP explanation failed:", e)

# --- Save metrics and produce a quick plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))
plt.plot(y_test_inv, label="Actual")
plt.plot(mu_inv, label="Predicted mean")
plt.fill_between(np.arange(len(mu_inv)).astype(int),
                 (mu_inv - 1.96*sigma_inv).reshape(-1),
                 (mu_inv + 1.96*sigma_inv).reshape(-1),
                 color='gray', alpha=0.3, label="95% PI")
plt.legend()
plt.title("Probabilistic forecasts vs actual")
plt.savefig("forecast_with_intervals.png")
print("Saved forecast plot as forecast_with_intervals.png")
