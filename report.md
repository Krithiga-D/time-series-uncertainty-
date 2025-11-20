# Project Report
**Advanced Time Series Forecasting with Neural Networks and Uncertainty Quantification**

## 1. Objective
Implement a probabilistic sequence forecasting model (LSTM) that provides both point forecasts and uncertainty estimates (prediction intervals). Compare performance vs classic baselines (ETS, ARIMA) using deterministic and probabilistic metrics and evaluate through rolling-origin cross-validation.

## 2. Dataset
- Synthetic multivariate dataset generated with `data_gen.py`.
- 2000 hourly observations by default; contains clear trend, daily & weekly seasonality, AR(1) behavior, spikes, and heteroscedastic noise.
- Features: `feature1`, `feature2`, `spikes` and `target`.
- Data generation documented in `data_gen.py`.

## 3. Feature engineering & preprocessing
- Standard scaling for features and target (to stabilize training).
- Sliding windows: 48-hour input (window_size=48) predicting next time-step (horizon=1).
- Rolling-origin cross-validation used (expanding training window) to measure generalization over time.

## 4. Model architecture
- Probabilistic LSTM outputs two numbers per sample: μ (mean) and log σ^2 (log variance).
- Loss: Gaussian negative log-likelihood (NLL) which encourages correct mean & variance learning.
- Dropout layers included; MC Dropout option available for alternative uncertainty estimation.
- Hyperparameters tuned via Optuna: number of LSTM units, dropout, layers, learning rate, and batch size.

## 5. Baselines
- Exponential Smoothing (ETS).
- ARIMA (simple order chosen; can be grid-searched for better performance).

## 6. Evaluation metrics
- Deterministic: RMSE, MAE, MAPE.
- Probabilistic: CRPS (Continuous Ranked Probability Score).
- Visual: predicted mean vs actual with 95% prediction intervals.

## 7. Cross-validation strategy
- Rolling-origin (expanding training window) with multiple folds (created in `src/train.py`).
- Optuna used across first few folds for efficient hyperparameter search; final model trained on full training portion.

## 8. Results (example outputs)
- Include final RMSE, MAE, MAPE, and CRPS for the probabilistic LSTM and for ETS/ARIMA baselines.
- Include `forecast_with_intervals.png` and `shap_summary.png` in report artifacts.

## 9. Interpretation of uncertainty
- Prediction intervals widen in regions of spikes or when model uncertainty increases (seen by larger σ predictions).
- CRPS quantifies how well predictive distribution matches observations — lower CRPS is better.

## 10. Limitations & Future Work
- Use of richer probabilistic families (Student-t) for heavy tails.
- Multivariate probabilistic model (joint covariance) instead of independent Gaussian per time-step.
- Increase Optuna trials and use more advanced Bayesian search for stronger performance.
