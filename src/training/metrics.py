import numpy as np
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluation_rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def evaluation_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def evaluation_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def evaluation_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)