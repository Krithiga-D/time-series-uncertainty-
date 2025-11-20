# Advanced Time Series Forecasting with Neural Networks and Uncertainty Quantification

This repo contains a Gold-level project implementing probabilistic time series forecasting:
- Synthetic dataset generator: `data_gen.py`
- Full pipeline in `src/`:
  - `train.py` (training, rolling-origin CV, Optuna tuning)
  - `models.py` (probabilistic LSTM)
  - `baseline.py` (ETS, ARIMA)
  - `evaluate.py` (RMSE, MAE, MAPE, CRPS)
  - `explainability.py` (SHAP)
- Report: `report.md`

## Quick start
1. (Optional) create a Python virtualenv
2. `pip install -r requirements.txt`
3. `python data_gen.py`  # generates data/simulated_timeseries.csv (2000 rows by default)
4. `python src/train.py` # trains model, runs CV, saves artifacts

