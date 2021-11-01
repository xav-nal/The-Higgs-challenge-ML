#!/usr/bin/env python
# coding: utf-8



### Load the training data into feature matrix, class labels, and event ids:
from proj1_helpers import *
DATA_TRAIN_PATH = 'data/train.csv.zip'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)


### Data cleaning and normalisation

# we know that very low values are used to signal unavailable data
tX[tX <= -999] = np.nan

# split feature 22 by value as planned
tX = np.hstack((np.delete(tX, 22,axis=1), np.stack([tX[:,22] == 0, tX[:,22] == 1, tX[:,22] == 2, tX[:,22] == 3]).T))


tX_mean = np.nanmean(tX,axis=0)
tX_std = np.nanstd(tX,axis=0)
norm_tX = np.subtract(tX, tX_mean, where=np.isfinite(tX_mean))
norm_tX = np.divide(norm_tX, tX_std, where=tX_std>0)

# now replace NaNs with the mean
norm_tX[np.isnan(norm_tX)] =0

# outliers: we remove any datapoint that is more than 4 standard derivations removed from the mean
# since normalisation was applied, this translate to
outliers_mask = np.linalg.norm(norm_tX, ord=np.inf, axis=1) > 4
norm_tX = np.delete(norm_tX, outliers_mask, axis=0)
y = np.delete(y, outliers_mask, axis=0)

### Ridge regression with a polynome 7 degree

from Implementation import split_data, build_poly, ridge_regression, compute_mse

def ridge_regression_split(x, y):
    """ridge regression demo."""
   
    # define parameter
    degree = 7
    seed = 29
    ratio = 0.8
    lambda_ = 0.0001
     
    # split data
    x_tr, x_te, y_tr, y_te = split_data(x, y, ratio, seed)
     
    # form tx
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)
    
    # ridge regression with different lambda
    rmse_tr = []
    rmse_te = []
    
    # ridge regression
    
    weight, loss_rmse = ridge_regression(y_tr, tx_tr, lambda_)
    rmse_tr.append(np.sqrt(2 * compute_mse(y_tr, tx_tr, weight)))
    rmse_te.append(np.sqrt(2 * compute_mse(y_te, tx_te, weight)))

    print("proportion={p}, degree={d}, lambda={l:.3f}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(
           p=ratio, d=degree, l=lambda_, tr=rmse_tr[0], te=rmse_te[0]))
    
    return weight, loss_rmse




weights, loss_rmse = ridge_regression_split(norm_tX, y)


### Generate predictions and save ouput in csv format for submission:
DATA_TEST_PATH = 'data/test.csv.zip'
y, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

## we apply the same normalisation to the test data than we did training
# treat NaNs
tX_test[tX_test <= -999] = np.nan
# split feature 22 by value as planned
tX_test = np.hstack((np.delete(tX_test, 22,axis=1), np.stack([tX_test[:,22] == 0, tX_test[:,22] == 1, tX_test[:,22] == 2, tX_test[:,22] == 3]).T))
# normalise
norm_tX_test = np.subtract(tX_test, tX_mean, where=np.isfinite(tX_mean))
norm_tX_test = np.divide(norm_tX_test, tX_std, where=tX_std>0)
# and replace NaNs with the mean
norm_tX_test[np.isnan(norm_tX_test)] =0

# and add polynomial features
norm_tX_test = build_poly(norm_tX_test, 7)

OUTPUT_PATH = 'data/sample-submission'
y_pred = predict_labels(weights, norm_tX_test)

create_csv_submission(ids_test, y_pred, OUTPUT_PATH)






