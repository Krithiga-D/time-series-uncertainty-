# src/models.py
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def nll_loss_gaussian(y_true, y_pred):
    """
    y_pred contains two outputs concatenated: mu and log_var
    We compute NLL for Gaussian: 0.5*(log_var + (y_true - mu)^2 / var) + constant
    """
    mu = y_pred[:, 0:1]
    log_var = y_pred[:, 1:2]
    var = K.exp(log_var) + 1e-6
    nll = 0.5 * (log_var + K.square(y_true - mu) / var)
    return K.mean(nll)

def build_probabilistic_lstm(window_size, n_features, units=64, dropout=0.2, lstm_layers=2, use_mc_dropout=False):
    inp = Input(shape=(window_size, n_features))
    x = inp
    for i in range(lstm_layers):
        return_seq = (i < lstm_layers - 1)
        x = LSTM(units, return_sequences=return_seq)(x)
        if dropout > 0:
            x = Dropout(dropout)(x, training=use_mc_dropout)  # if use_mc_dropout True, keep dropout active at inference
    # output two parameters: mu and log_var
    mu = Dense(1, name="mu")(x)
    log_var = Dense(1, name="log_var")(x)
    out = tf.keras.layers.Concatenate(axis=1)([mu, log_var])  # shape (batch, 2)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss=nll_loss_gaussian, metrics=["mae"])
    return model

# Utility wrappers to extract mu & sigma
def predict_mu_sigma(model, X):
    """
    Given trained model producing [mu, log_var], return mu, sigma
    """
    preds = model.predict(X)
    mu = preds[:, 0:1]
    log_var = preds[:, 1:2]
    sigma = np.sqrt(np.exp(log_var) + 1e-6)
    return mu, sigma
