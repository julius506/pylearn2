# -*- coding: utf-8 -*-
#Original author Ian Goodfellow
#https://blog.safaribooksonline.com/2014/02/10/pylearn2-regression-3rd-party-data/

# We'll need the csv module to read the file
import csv
# We'll need numpy to manage arrays of data
import numpy as np

# We'll need the DenseDesignMatrix class to return the data
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

def load_data(start, stop):
    """
    SAheart2 with the first column containing the target data:

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
    with open('SAheart2.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        X = []
        y = []
        header = False
        for row in reader:
            # Skip the first row containing the string names of each attribute
            if header:
                header = True
                continue
            # Convert the row into numbers
            row = [float(elem) for elem in row]
            X.append(row[2:])
            y.append(row[1])
    X = np.asarray(X)
    y = np.asarray(y)
    y = y.reshape(y.shape[0], 1)

    X = X[start:stop, :]
    y = y[start:stop, :]

    return DenseDesignMatrix(X=X, y=y)
