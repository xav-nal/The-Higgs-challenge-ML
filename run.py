import numpy as np
from helpers import split_dataset
from implementation import *
from costs import compute_mse, compute_mse_loss
from helpers import build_poly

def commun_run(y,x,
    ratio = 0.8,
    lambda_ = 0.0001,
    seed = 29,
    poly = False,
    degree = 7):

    x_tr, x_te, y_tr, y_te = split_dataset(x, y, ratio=0.8, seed=seed)

    if poly:
        x_tr = build_poly(x_tr, degree)
        x_te = build_poly(x_te, degree)

    return x_tr, x_te, y_tr, y_te
        
    

def run_ridge_regression(y, x,
    degree = 7,
    ratio = 0.8,
    lambda_ = 0.0001,
    seed = 29):
    """Run the Ridge Regression algorithm on the data (x,y)"""

    tx_tr, tx_te, y_tr, y_te = commun_run(y, x, ratio=ratio, seed=seed, poly=True)
   
    
    # ridge regression
    weight, loss_rmse = ridge_regression(y_tr, tx_tr, lambda_)
    rmse_tr = np.sqrt(2 * compute_mse_loss(y_tr, tx_tr, weight))
    rmse_te = np.sqrt(2 * compute_mse_loss(y_te, tx_te, weight))

    print("proportion={p}, degree={d}, lambda={l:.3f}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(
           p=ratio, d=degree, l=lambda_, tr=rmse_tr, te=rmse_te))
    
    return weight, loss_rmse
    

def run_logistic_regression(y, x,
    ratio = 0.8,
    max_iters = 1000,
    gamma = 1e-5):
    """Run the Logistic Regression algorithm on the data (x,y)"""
    

    tx_tr, tx_te, y_tr, y_te = commun_run(y, x, ratio=ratio, poly=True)
    print("tx_tr", tx_tr.shape)
    
    # logistic regression
    D = tx_tr.shape[1]
    initial_w = np.zeros((D, 1))
    weight, loss = logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma)

    loss_tr = compute_log_likelihood(y_tr, tx_tr, weight)
    loss_te = compute_log_likelihood(y_te, tx_te, weight)


    print("proportion={p}, max_iters={i}, gamma={g:.3f}, Training Loss={tr:.3f}, Testing Loss={te:.3f}".format(
           p=ratio, i=max_iters, g=gamma, tr=loss_tr.item(), te=loss_te.item()))
    
    return weight, loss