# src/baseline.py
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

def fit_ets(train_series, seasonal_periods=24, trend='add', seasonal='add'):
    model = ExponentialSmoothing(train_series, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    fit = model.fit(optimized=True)
    return fit

def fit_arima(train_series, order=(2,0,2)):
    model = ARIMA(train_series, order=order)
    fit = model.fit()
    return fit

def forecast_model(fit_model, steps):
    # statsmodels results have forecast or predict method
    try:
        fcasts = fit_model.forecast(steps)
    except Exception:
        fcasts = fit_model.predict(start=len(fit_model.model.endog), end=len(fit_model.model.endog)+steps-1)
    return np.array(fcasts)
