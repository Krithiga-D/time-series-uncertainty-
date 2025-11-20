# src/evaluate.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# For CRPS we'll use properscoring.crps_gaussian
from properscoring import crps_gaussian

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100

def crps_score(y_true, mu, sigma):
    """
    y_true: shape (n,1) or (n,)
    mu, sigma: arrays of same shape representing Gaussian forecast
    returns average CRPS
    """
    y_true = np.array(y_true).reshape(-1)
    mu = np.array(mu).reshape(-1)
    sigma = np.array(sigma).reshape(-1)
    crps_vals = crps_gaussian(y_true, mu, sigma)
    return np.mean(crps_vals)
