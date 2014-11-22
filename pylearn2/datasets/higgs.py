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
    Loads the smallHiggs dataset (First 50000 samples) from:


    Baldi, P., P. Sadowski, and D. Whiteson. “Searching for Exotic Particles in High-energy Physics with Deep Learning.” Nature Communications 5 (July 2, 2014).
    https://archive.ics.uci.edu/ml/datasets/HIGGS#

    The data has been produced using Monte Carlo simulations. The first 21 features (columns 2-22) are kinematic properties measured by the particle detectors in the accelerator. The last seven features are functions of the first 21 features; these are high-level features derived by physicists to help discriminate between the two classes. There is an interest in using deep learning methods to obviate the need for physicists to manually develop such features. Benchmark results using Bayesian Decision Trees from a standard physics package and 5-layer neural networks are presented in the original paper. The last 500,000 examples are used as a test set.

    The first column is the class label (1 for signal, 0 for background), followed by the 28 features (21 low-level features then 7
high-level features): lepton pT, lepton eta, lepton phi, missing energy magnitude, missing energy phi, jet 1 pt, jet 1 eta, jet 1 phi, jet 1 b-tag, jet 2 pt, jet 2 eta, jet 2 phi, jet 2 b-tag, jet 3 pt, jet 3 eta, jet 3 phi, jet 3 b-tag, jet 4 pt, jet 4 eta, jet 4 phi, jet 4 b-tag, m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb. For more detailed information about each feature see the original paper.

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
    with open('smallHIGGS.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        X = []
        y = []
        header = True
        for row in reader:
            # Skip the first row containing the string names of each attribute
            if header:
                header = False
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
