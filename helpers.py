import numpy as np
import zipfile

def sigmoid(t):
    """Apply sigmoid function on t
    """
    # t = np.clip(t, -500, 500)
    # return 1.0 / (1 + np.exp(-t))
    return np.exp(-np.logaddexp(0, -t))

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    with zipfile.ZipFile(data_path) as ar, ar.open(ar.infolist()[0]) as f:
        y = np.genfromtxt(f, delimiter=",", skip_header=1, dtype=str, usecols=1)
        with ar.open(ar.infolist()[0]) as f:
            x = np.genfromtxt(f, delimiter=",", skip_header=1)

    ids = x[:,0].astype(int)
    input_data = x[:,2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[y=='b'] = -1 #  old implementation yb[np.where(y=='b')]

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids



        

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    #print(np.shape(x))
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def split_dataset(x, y, ratio, seed=1):
    """Split the dataset based on the split ratio."""
    np.random.seed(seed)
    
    N = len(y)
    indices = np.random.permutation(N)
    
    idx_split = int(np.floor(N * ratio))
    idx_tr, idx_te = indices[:idx_split], indices[idx_split:]

    x_tr, x_te = x[idx_tr], x[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    
    return x_tr, x_te, y_tr, y_te



def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
    