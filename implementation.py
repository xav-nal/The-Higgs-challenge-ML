import numpy as np
from costs import compute_mse, compute_mse_loss, update_gamma, compute_log_likelihood
from helpers import batch_iter, sigmoid

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent
    D is the number of features.
    N is the number of samples.

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N,D).

    Returns:
        w: last weight vector, numpy array of shape (D,)
        loss: loss value (cost function)
    """
    loss = 0
    w = initial_w
    N = len(y)

    for _ in range(max_iters):
        y_pred = np.dot(tx, w)
        err = y - y_pred
        loss = compute_mse(err)
        
        grad = - tx.T.dot(err) / N
        w = w - gamma * gradient

    return w, loss
        

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent
    D is the number of features.
    N is the number of samples.

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N,D).

    Returns:
        w: last weight vector, numpy array of shape (D,)
        loss: loss value (cost function)
    """
    loss = 0
    w = initial_w
    N = len(y)
    batch_size = N / 10

    for y_batch, tx_batch in batch_iter(
            y, tx, batch_size=batch_size, num_batches=1
        ):
            err = y - tx.dot(w)
            grad = -tx.T.dot(err) / N
            w = w - gamma * grad
            # calculate loss
            loss = compute_mse(err)

    return w, loss

def least_squares(y, tx):
    """Least squares regression using normal equations
    D is the number of features.
    N is the number of samples.

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N,D).

    Returns:
        w: last weight vector, numpy array of shape (D,)
        loss: loss value (cost function)
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_mse_loss(y, tx, w)
    return w, mse

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations
    D is the number of features.
    N is the number of samples.

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N,D).
        lambda_: scalar.

    Returns:
        w: last weight vector, numpy array of shape (D,)
        loss: loss value (cost function)
    """
    N = tx.shape[0]
    D = tx.shape[1]
    aI = 2 * N * lambda_ * np.identity(D)
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = np.sqrt(2 * compute_mse_loss(y, tx, w))
        
    return w, loss  

def logistic_regression(y, X, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD (y \in {0,1})
    D is the number of features.
    N is the number of samples.

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N,D).
        initial_w: numpy array of shape (D,).
        max_iters: scalar.
        gamma: scalar.

    Returns:
        w: last weight vector, numpy array of shape (D,)
        loss: loss value (cost function)
    """
    assert y.shape[0] == X.shape[0]
    # assert X.shape[1] == initial_w.shape[0]
    y = y.reshape(-1,1)
    
    
    N = X.shape[0]
    loss = 0
    losses = []
    w = np.zeros((X.shape[1], 1))
    threshold = 1e-8
    for iteration in range(max_iters):
        pred = sigmoid(X.dot(w))
        err = pred -y
        loss = compute_log_likelihood(y, X, w)
        grad = X.T.dot(err) / N
        w = w - gamma * grad


        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

        if (iteration + 1) % 100 == 0:
            print(f"Iteration [{iteration + 1}/{max_iters}], Loss [{loss}]")
    
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent or SGD
    D is the number of features.
    N is the number of samples.

    Args:
        y: numpy array of shape (N,).
        tx: numpy array of shape (N,D).
        lambda_: scalar.
        initial_w: numpy array of shape (D,).
        max_iters: scalar.
        gamma: scalar.

    Returns:
        w: last weight vector, numpy array of shape (D,)
        loss: loss value (cost function)
    """
    threshold = 1e-8
    losses = []
    w = initial_w
    N = tx.shape[0]

    for _ in range(max_iters):
        pred = sigmoid(tx.dot(w))
        err = pred -y
        loss = compute_log_likelihood(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        grad = -tx.T.dot(err) / N + 2 * lambda_ * w
        w = w - gamma * grad


        # update gamme
        gamma = update_gamma(gamma, loss)

        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        
    return w, loss