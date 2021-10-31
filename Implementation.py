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
#logistic regretion helper
#-----------------------------------------

def update_gamma(gamma,loss):
    """update the gamme during the iteration phase"""
    if(loss < 20000):
        if(loss < 7000):
            gamma = 0.0000000005
            if(loss < 1000):
                gamma = 0.0000000001
                if(loss < 100):
                    gamma = 0.00000000001
                    if(loss < 40):
                        gamma = 0.000000000009
                        
                        
        else:
            gamma = 0.000000001
                
    return gamma

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))
    

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    #print(loss)
    return np.squeeze(-loss)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
 
    pred = sigmoid(tx.dot(w))
    
    #print(pred)
    
    grad = tx.T.dot(pred - y)
    #print('le gradient')
    
    return grad


def learning_by_gd(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w -= gamma * grad
    return loss, w, grad
#-----------------------------------------
#helpers function for penalized logistic regression
#-----------------------------------------

def learning_by_penalized_gd(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    #print(loss)
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    #loss, gradient = penalized_lr(y, tx, w, lambda_)
    w -= gamma * gradient
    
    return loss, w, gradient



#-----------------------------------------
#least squares GD(y, tx, initial w,max iters, gamma)
#Linear regression using gradient descent
#-----------------------------------------

#-----------------------------------------
#least squares SGD(y, tx, initial w,max iters, gamma)
#Linear regression using stochastic gradient descent
#-----------------------------------------

#-----------------------------------------
#least squares(y, tx)
#Least squares regression using normal equations
#-----------------------------------------

def least_squares(y, tx):
    """calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)  
    return np.linalg.solve(a, b)

#-----------------------------------------
#ridge regression(y, tx, lambda )
#Ridge regression using normal equations
#-----------------------------------------

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
        loss, w, gradient = learning_by_gd(y, tx, w, gamma)
        
        gamma = update_gamma(gamma,loss)
        
        # info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}, gamma={g}".format(i=iter, l=loss, g=gamma))
            
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
   
    #print final loss
    print("loss={l}".format(l=calculate_loss(y, tx, w))) 
    
    return w

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
    
    return w
#-----------------------------------------