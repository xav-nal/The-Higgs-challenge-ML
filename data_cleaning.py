import numpy as np


def clean_dataset(y,x):
    """Clean and standardize the data
    """
    x[x <= -999] = np.nan

    x = np.hstack((np.delete(x, 22, axis=1),
                np.array(x[:,22]==0).reshape(-1,1),
                np.array(x[:,22]==1).reshape(-1,1), 
                np.array(x[:,22]==2).reshape(-1,1),
                np.array(x[:,22]==3).reshape(-1,1)))


    mean_x = np.nanmean(x, axis=0)
    centered_x = x - mean_x
    centered_x = np.where(np.isnan(centered_x), mean_x, centered_x)
    std_x = np.nanstd(centered_x, axis=0)
    data_x = centered_x / std_x

    outliers_mask = np.linalg.norm(data_x, ord=np.inf, axis=1) > 4
    final_x = np.delete(data_x, outliers_mask, axis=0)
    y = np.delete(y, outliers_mask, axis=0)

    return y, final_x
    
