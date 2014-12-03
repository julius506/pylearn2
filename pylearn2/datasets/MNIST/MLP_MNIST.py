import os
import sys
import pylearn2
import numpy as np
import csv
import pandas as pd
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter
from pylearn2 import *
from pylearn2.train_extensions import best_params
from pylearn2 import train
#import load_data

sys.path.append(os.path.abspath("~/projects/pylearn2/pylearn2/pylearn2/scripts"))
sys.path.append(os.path.abspath("~/projects/pylearn2/pylearn2/datasets/MNIST"))
sys.path.append(os.path.abspath("~/projects/pylearn2/pylearn2"))




import numpy as np

from pandas import read_csv

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

def load_data(start, stop, classes=2, path=''):
    """
    Expects
    -------
    valid .csv file in the same folder
    First classes columns must be at the beggining in a binary format. For example, for 2 classes the first
    column must have zeros when 
    
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
       
    data = read_csv(path, sep=",")
    
    X = data.ix[start:stop, (classes):]
    y = data.ix[start:stop, 0:(classes)]
    X = np.array( X.as_matrix() )
    y = np.array( y.as_matrix() )

    return DenseDesignMatrix(X=X, y=y)











# Load the datasets
print 'Loading the datasets...'

trainingSet = load_data(start=0,stop=50000, classes=10, path='./mnist_train2.csv')
validationSet = load_data(start=50000,stop=60000, classes=10, path='./mnist_train2.csv')
testSet = load_data(start=0,stop=10000, classes=10, path='./mnist_test2.csv')
data = {'train': trainingSet, 'valid':validationSet, 'test': testSet}



# Define layers
print 'Defining the layers...'
hidden_layer1 = mlp.Sigmoid(layer_name='h0', dim=1000, irange=.1, init_bias=1.)
hidden_layer2 = mlp.Sigmoid(layer_name='h0', dim=1000, irange=.1, init_bias=1.)
output_layer = mlp.Softmax(n_classes=10, layer_name='output', irange=.1)

layers1 = [hidden_layer1, hidden_layer2, output_layer]


# Define termination criteria
print 'Defining the termination criteria...'
termination_criterion1 = termination_criteria.MonitorBased(channel_name='valid_output_misclass',
                                                           N=10, 
                                                           prop_decrease=0.0)
termination_criterion2 = termination_criteria.EpochCounter(1000)
termination_criterionList = termination_criteria.And(criteria=[termination_criterion1, 
                                                               termination_criterion2])



# Training algorithm
print 'Initilizing the training algorithm...'
trainAlgorithm = training_algorithms.sgd.SGD(learning_rate=.01, 
                         batch_size=10, 
                         termination_criterion=termination_criterionList,
                         monitoring_dataset=data)



# Define extensions
print 'Defining the extensions...'
monitor_save_best = best_params.MonitorBasedSaveBest(channel_name='valid_output_misclass',
                                                     save_path='./MLP_best.pkl')
extensions = [monitor_save_best]


# Initilize the model
print 'Initializing the model...'
model1 = mlp.MLP(layers=layers1, nvis=784)

# Initialize the train object
print 'Initializing the train object...'
train = pylearn2.train.Train(
                             dataset = trainingSet,
                             model = model1,
                             extensions = extensions,
                             algorithm = trainAlgorithm,
                             save_path = './MLP.pkl',
                             save_freq = 1)

# Train
print 'Starting the training loop...'
train.main_loop()




