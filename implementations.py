import numpy as np

#-----------------------------------------
#Polynomial regression
#-----------------------------------------

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def compute_mse(y, tx, w):
    """compute the loss by mse."""
           
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def polynomial_regression(x):
    """Constructing the polynomial basis function expansion of the data,
       and then running least squares regression, and returning the weight."""
    # define parameters
    degrees = [1,3,7,12]
    
    
    for ind, degree in enumerate(degrees):
        # form dataset to do polynomial regression.
        tx = build_poly(x, degree)
        
        # least squares
        weight = least_squares(y, tx)
        
        # compute RMSE
        rmse = np.sqrt(2 * compute_mse(y, tx, weights))
        print("Processing {i}th experiment, degree={d}, rmse={loss}".format(
              i=ind + 1, d=degree, loss=rmse))
       
        
    return weight
#-----------------------------------------
#split Log Reg
#-----------------------------------------
def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    
    return x_tr, x_te, y_tr, y_te

def train_test_split_demo_lr(x, y, ratio, seed):
    """polynomial regression with different split ratios and different degrees."""
    x_tr, x_te, y_tr, y_te = split_data(x, y, ratio, seed)
    
    # form tx 
    tx_tr =x_tr
    tx_te = x_te
    
    # init parameters
    max_iter = 1000
    gamma = 0.00000009
    initial_w = np.zeros((tx_tr.shape[1], 1))
    
    
    weight = logistic_regression_gd(y_tr, tx_tr, initial_w, max_iter, gamma)
    
    y_tr = np.expand_dims(y_tr, axis=1)
    y_te = np.expand_dims(y_te, axis=1)
    
    #calculate cost for train and test data
    cost_tr = calculate_loss(y_tr, tx_tr, weight)
    cost_te = calculate_loss(y_te, tx_te, weight)

    print("proportion={p}, logistic reg, Training loss={tr:.3f}, Testing loss={te:.3f}".format(
          p=ratio, tr=cost_tr, te=cost_te))
    return weight

#-----------------------------------------
#logistic regretion helper
#-----------------------------------------

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

def sigmoid(t):
    """apply sigmoid function on t."""
    #return 1.0 / (1 + np.exp(-t))
    return np.exp(-np.logaddexp(0, -t))

    
    

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    y_pred = sigmoid(tx.dot(w))
    loss = 0
    loss = y.T.dot(np.log(y_pred)) + (1 - y).T.dot(np.log(1 - y_pred))
    
    return np.squeeze(-loss)
    

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
 
    y_pred = sigmoid(tx.dot(w))
    
    grad = tx.T.dot(y_pred - y)
    
    return grad

def compute_gradient_LR(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad


def learning_by_gd(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    #grad = calculate_gradient(y, tx, w)
    grad = compute_stoch_gradient(y, tx, w)
    w -= gamma * grad
    return loss, w
#-----------------------------------------
#helpers function for penalized logistic regression
#-----------------------------------------

def learning_by_penalized_gd(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    #loss, gradient = penalized_lr(y, tx, w, lambda_)
    w -= gamma * gradient
    
    return loss, w, gradient



#-----------------------------------------
#least squares GD(y, tx, initial w,max iters, gamma)
#Linear regression using gradient descent
#-----------------------------------------
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Gradient descent algorithm  """
    loss = 0
    
    # start the linear regression
    for iter in range(max_iter):
        gradient, err = compute_gradient_LR(y, tx, w)
        
        loss = compute_mse(y, tx, w)
        
        #update weights
        w -= gamma*gradient
        
    
    return w,loss
    
#-----------------------------------------
#least squares SGD(y, tx, initial w,max iters, gamma)
#Linear regression using stochastic gradient descent
#-----------------------------------------
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """ Stochastic gradient descent algorithm  """
    loss = 0
    
    # start the linear regression
    for iter in range(max_iter):
        gradient, err = compute_stoch_gradient(y, tx, w)
        
        loss = compute_mse(y, tx, w)
        
        #update weights
        w -= gamma*gradient
        
    
    return w,loss

#-----------------------------------------
#least squares(y, tx)
#Least squares regression using normal equations
#-----------------------------------------

def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)  
    w_sol = np.linalg.solve(a, b)
    
    loss = compute_mse(y, tx, w_sol)
    
    return w_sol,loss

#-----------------------------------------
#ridge regression(y, tx, lambda )
#Ridge regression using normal equations
#-----------------------------------------
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    n = tx.shape[0]
    #compute lambda prime
    lambda_prime = 2 * n * lambda_
    
    #normal equation aw = b
    a = tx.T.dot(tx) + lambda_prime* np.identity(tx.shape[1])
    b = tx.T.dot(y)
    
    weight = np.linalg.solve(a, b)
    
    loss_rmse = np.sqrt(2 * compute_mse(y, tx, weight))
    
    return weight, loss_rmse  
    
#-----------------------------------------
#logistic regression(y, tx, initial w,max iters, gamma)
#Logistic regression using gradient descent or SGD
#-----------------------------------------
def logistic_regression_gd(y, tx, initial_w, max_iter, gamma):
    """ logistic regression function"""
    #log reg parameters
    losses = []
    threshold = 1e-8
    
    
    # build array
    w = initial_w
    y = np.expand_dims(y, axis=1)
   
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gd(y, tx, w, gamma)
        
        #gamma = update_gamma(gamma,loss)
        
        
        
        # info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}, gamma={g}".format(i=iter, l=loss, g=gamma))
            
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
   
    #print final loss
    print("loss={l}".format(l=calculate_loss(y, tx, w))) 
    
    return w, loss

#-----------------------------------------
#reg logistic regression(y, tx, lambda ,initial w, max iters, gamma)
#Regularized logistic regression using gradient descent or SGD
#-----------------------------------------
                
def regu_logistic_regression_gd(y, tx, lambda_, initial_w, max_iter, gamma ):
    """regu logistic regression gd function"""
    # rlr parameters
    threshold = 1e-8
    losses_rlr = []

    # define array
    w = initial_w
    y = np.expand_dims(y, axis=1)

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w, gradient = learning_by_penalized_gd(y, tx, w, gamma, lambda_)
        
        gamma = update_gamma(gamma,loss)
        
        # info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses_rlr.append(loss)
        if len(losses_rlr) > 1 and np.abs(losses_rlr[-1] - losses_rlr[-2]) < threshold:
            break
    
    print("loss={l}".format(l=calculate_loss(y, tx, w)))
    
    return w, loss
#-----------------------------------------
