# We'll need the csv module to read the file
import csv
# We'll need numpy to manage arrays of data
import numpy as np

# We'll need the DenseDesignMatrix class to return the data
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

def load_data(start, stop):
    """
    Loads the red wine quality dataset from:

    Concrete Compressive Strength 

---------------------------------

Data Type: multivariate
 
Abstract: Concrete is the most important material in civil engineering. The 
concrete compressive strength is a highly nonlinear function of age and 
ingredients. These ingredients include cement, blast furnace slag, fly ash, 
water, superplasticizer, coarse aggregate, and fine aggregate.

---------------------------------

Sources: 

  Original Owner and Donor
  Prof. I-Cheng Yeh
  Department of Information Management 
  Chung-Hua University, 
  Hsin Chu, Taiwan 30067, R.O.C.
  e-mail:icyeh@chu.edu.tw
  TEL:886-3-5186511

  Date Donated: August 3, 2007
 
---------------------------------

Data Characteristics:
    
The actual concrete compressive strength (MPa) for a given mixture under a 
specific age (days) was determined from laboratory. Data is in raw form (not scaled). 

Summary Statistics: 

Number of instances (observations): 1030
Number of Attributes: 9
Attribute breakdown: 8 quantitative input variables, and 1 quantitative output variable
Missing Attribute Values: None

---------------------------------

Variable Information:

Given is the variable name, variable type, the measurement unit and a brief description. 
The concrete compressive strength is the regression problem. The order of this listing 
corresponds to the order of numerals along the rows of the database. 

Name -- Data Type -- Measurement -- Description

Cement (component 1) -- quantitative -- kg in a m3 mixture -- Input Variable
Blast Furnace Slag (component 2) -- quantitative -- kg in a m3 mixture -- Input Variable
Fly Ash (component 3) -- quantitative -- kg in a m3 mixture -- Input Variable
Water (component 4) -- quantitative -- kg in a m3 mixture -- Input Variable
Superplasticizer (component 5) -- quantitative -- kg in a m3 mixture -- Input Variable
Coarse Aggregate (component 6) -- quantitative -- kg in a m3 mixture -- Input Variable
Fine Aggregate (component 7) -- quantitative -- kg in a m3 mixture -- Input Variable
Age -- quantitative -- Day (1~365) -- Input Variable
Concrete compressive strength -- quantitative -- MPa -- Output Variable 
---------------------------------

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
    with open('winequality_red.csv', 'r') as f:
        reader = csv.reader(f, delimiter=';')
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
            X.append(row[:-1])
            y.append(row[-1])
    X = np.asarray(X)
    y = np.asarray(y)
    y = y.reshape(y.shape[0], 1)

    X = X[start:stop, :]
    y = y[start:stop, :]

    return DenseDesignMatrix(X=X, y=y)
