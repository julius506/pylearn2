import pylearn2
import numpy as np
import pandas
from pandas import read_csv
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

def load_data(start, stop):
    """
    Expects
    -------
    valid .csv file in the same folder
    first column must be target
    Target must be 
    
    Parameters
    ----------
    start: int
    stop: int

    Returns
    -------

    dataset : DenseDesignMatrix
        A dataset include examples start (inclusive) through stop (exclusive).
        The start and stop parameters are useful for splitting the data into
        train, validation, and test data.
    """
            
    data = read_csv('smallHIGGS.csv', sep=",")
        
    X = data.ix[start:stop, 1:]
    y = data.ix[start:stop, 0]
            
    X = np.array( X.as_matrix() )
    y = np.array( y.as_matrix() )
    y = np.vstack([y,abs(y-1)])
    y = y.transpose()
    y = y.astype(int)
    #print type(X[1,9])
    #print type(y[1,1])
    #print y

    return DenseDesignMatrix(X=X, y=y)
