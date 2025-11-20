# src/explainability.py
import shap
import numpy as np
import matplotlib.pyplot as plt

def explain_shap(model, X_train_sample, X_test_sample, feature_names=None, save_path="shap_summary.png"):
    """
    model: Keras model returning [mu, log_var] - SHAP needs function that maps X->mu (predict mean).
    """
    # Create a wrapper predict function that returns mean
    def predict_mu(x):
        preds = model.predict(x)
        mu = preds[:, 0:1]
        return mu

    # Use KernelExplainer if DeepExplainer gives issues for TF versions
    try:
        explainer = shap.DeepExplainer(model, X_train_sample)
        shap_values = explainer.shap_values(X_test_sample)
        shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values
    except Exception:
        explainer = shap.KernelExplainer(predict_mu, X_train_sample[:100])
        shap_vals = explainer.shap_values(X_test_sample[:50], nsamples=200)

    # shap_vals shape: (n_samples, timesteps, features) -> aggregate by feature across timesteps (sum or mean)
    shap_vals_arr = np.array(shap_vals)
    # For summary plot, flatten time dimension
    n, t, f = shap_vals_arr.shape
    shap_vals_flat = shap_vals_arr.reshape(n, t*f)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(t*f)]
    shap.summary_plot(shap_vals_flat, features=X_test_sample.reshape(n, t*f), feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(save_path)
    return save_path
