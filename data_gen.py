# data_gen.py
"""
Generates a multivariate time series dataset with trend, multiple seasonalities,
autoregressive components, and noise. Saves to data/simulated_timeseries.csv
Default size: 2000 hourly observations (about 83 days) — adjust n_steps as needed.
"""
import numpy as np
import pandas as pd
import os

def generate(n_steps=2000, seed=42, out_path="data/simulated_timeseries.csv"):
    np.random.seed(seed)
    t = np.arange(n_steps)

    # Trend component
    trend = 0.0008 * t

    # Seasonalities: daily and weekly (if hourly data)
    daily = 2.5 * np.sin(2 * np.pi * t / 24)
    weekly = 1.8 * np.sin(2 * np.pi * t / (24*7))

    # Feature1: base temperature-like signal with trend + seasonality + AR(1)
    ar1 = np.zeros(n_steps)
    rho = 0.6
    noise_ar = np.random.normal(scale=0.5, size=n_steps)
    for i in range(1, n_steps):
        ar1[i] = rho * ar1[i-1] + noise_ar[i]

    feature1 = 20 + trend + daily + weekly + 0.5*ar1

    # Feature2: random-walk / slow-moving process (cumulative small noise)
    rw_noise = np.random.normal(scale=0.2, size=n_steps)
    feature2 = 50 + np.cumsum(rw_noise) * 0.05

    # Feature3: exogenous periodic binary events (spikes)
    spikes = (np.random.rand(n_steps) < 0.01).astype(float) * np.random.uniform(3,7,size=n_steps)

    # target is non-linear combination + heteroscedastic noise
    base = 0.6*feature1 - 0.3*feature2*0.02 + spikes
    # heteroscedastic noise: variance grows with base magnitude
    eps = np.random.normal(scale=0.2 + 0.05*np.abs(base), size=n_steps)
    target = base + eps

    df = pd.DataFrame({
        "time": pd.date_range(start="2020-01-01", periods=n_steps, freq="H"),
        "feature1": feature1,
        "feature2": feature2,
        "spikes": spikes,
        "target": target
    })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} with shape {df.shape}")
    return df

if __name__ == "__main__":
    generate()
