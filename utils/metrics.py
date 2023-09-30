import os
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_log_error

def MAE(y_true, y_pred):
    """ Mean Absolute Error """
    # print(f'{y_true.shape = }')
    # print(f'{y_pred.shape = }')
    # print(f'{(y_true - y_pred).shape = }')
    # return np.mean(np.abs(y_true - y_pred))
    return mean_absolute_error(y_true=y_true, y_pred=y_pred)

def MSE(y_true, y_pred):
    """ Mean Squared Error """ 
    # return np.mean((y_true - y_pred) ** 2)
    return mean_squared_error(y_true=y_true, y_pred=y_pred)

def RMSE(y_true, y_pred):
    """ Root Mean Squared Error """
    return np.sqrt(np.mean((y_true-y_pred)**2))

def MSLE(y_true, y_pred):
    """ Mean Squared Logarithmic Error """ 
    return mean_squared_log_error(y_true=y_true, y_pred=y_pred)

def MAPE(y_true, y_pred):
    """ Mean Absolute Percentage Error """
    # return np.mean(np.abs((y_true-y_pred) / (y_true + 1e-10))) * 100
    return mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)

def sMAPE(y_true, y_pred):
    """ Symmetric Mean Absolute Percentage Error """
    return np.mean(np.abs(y_pred - y_true) / ((np.abs(y_pred) + np.abs(y_true))/2)) * 100 

def R2(y_true, y_pred):
    # return 1 - (np.sum(np.power(y - yhat, 2)) / np.sum(np.power(y - np.mean(y), 2)))
    # 1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))
    return r2_score(y_true, y_pred)

def numpy_normalised_quantile_loss(y_true, y_pred, quantile):
    """Computes normalised quantile loss for numpy arrays.

    Uses the q-Risk metric as defined in the "Training Procedure" section of the
    main TFT paper.

    Args:
    y_true: Targets
    y_pred: Predictions
    quantile: Quantile to use for loss calculations (between 0 & 1)

    Returns:
    Float for normalised quantile loss.
    """
    assert 0 <= quantile <= 1, "Number should be within the range [0, 1]." 
    prediction_underflow = y_true - y_pred
    weighted_errors = quantile * np.maximum(prediction_underflow, 0.) + (1. - quantile) * np.maximum(-prediction_underflow, 0.)
    quantile_loss = weighted_errors.mean()
    normaliser = abs(y_true).mean()

    return 2 * quantile_loss / normaliser

metric_dict = {
    'R2' : R2,
    'MAE' : MAE, 
    'MSE' : MSE,
    'RMSE' : RMSE,
    'MAPE' : MAPE, 
    'sMAPE' : sMAPE,
    'nQL.5': lambda y_true, y_pred: numpy_normalised_quantile_loss(y_true=y_true, y_pred=y_pred, quantile=0.5),
    'nQL.9': lambda y_true, y_pred: numpy_normalised_quantile_loss(y_true=y_true, y_pred=y_pred, quantile=0.9),
}

def score(y, 
          yhat, 
          r,
          scaler={}):
    if scaler != {}: 
        if scaler['method'] == 'minmax':
            y = y * (scaler['max'] - scaler['min']) + scaler['min']
            yhat = yhat * (scaler['max'] - scaler['min']) + scaler['min']
        elif scaler['method'] == 'standard': 
            y = y * scaler['std'] + scaler['mean']
            yhat = yhat * scaler['std'] + scaler['mean']
        elif scaler['method'] == 'robust':
            y = y * scaler['iqr'] + scaler['median']
            yhat = yhat * scaler['iqr'] + scaler['median']

    if len(yhat.shape) == 3: 
        nsamples, nx, ny = yhat.shape
        yhat = yhat.reshape((nsamples,nx*ny))

    y = np.squeeze(y)
    yhat = np.squeeze(yhat)
    
    print(f'{y.shape = }')
    print(f'{yhat.shape = }')

    if r != -1:
        results = [str(np.round(np.float64(metric_dict[key](y, yhat)), r)) for key in metric_dict.keys()]
    else:
        results = [str(metric_dict[key](y, yhat)) for key in metric_dict.keys()]    

    print(results)
    return results