import numpy as np
from helpers import sigmoid

def compute_mse(error):
    """Compute the Mean Squared Error (MSE)

    Args:
        error: np.array of shape (N,)
    """
    return 1 / 2 * np.mean(error**2)

def compute_mse_loss(y, tx, w):
    """Compute the Mean Squared Error (MSE) Loss

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N,D).
        w: numpy array of shape (D,)
    """
    e = y - tx.dot(w)
    return compute_mse(e)


def compute_log_likelihood(y, tx, w):
    """Compute the Log Likelihood
    
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negtive loss
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    N = y.shape[0]
    xw = tx.dot(w)
    pred =  sigmoid(xw)
    pred = np.clip(pred, 1e-15, 1 - 1e-15)
    
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    # print("loss: ",loss)
    # loss = np.sum(np.where(xw > 23, xw, np.log(1 + np.exp(xw))) - y * xw)
    # loss = np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    
    return loss
    

    
    # if (np.isnan(pred).sum()):
    #     print("pred", np.isnan(pred).sum())
    #     print("Loss",loss)
        
    # return np.squeeze(-loss).item() * (1 / N)





def update_gamma(gamma,loss):
    """update the gamme during the iteration phase"""
    if(loss < 20000):
        if(loss < 7000):
            gamma = 0.0000000005
            if(loss < 1000):
                gamma = 0.0000000003
                if(loss < 100):
                    gamma = 0.00000000001
                    if(loss < 40):
                        gamma = 0.000000000009      
                        
        else:
            gamma = 0.000000001
                
    return gamma


    

















    