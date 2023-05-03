import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv(r"Algorithm\excel\cost_average.csv")

""" Takes a file with data, and plots the data on with linear regression"""
def numpy_reg():
    # Gathers the arguments from the prompt
    n = 3

    X = np.array(df["Capacity"])
    Y = np.array(df["Cost"])

    # extracts the data from  the values and place in X,Y
    #X, Y = np.transpose(val)
    Xp  = powers(X,0,int(n))
    Yp  = powers(Y,1,1)
    Xpt = Xp.transpose()

    # Calcualtes the a values for each X
    a = np.matmul(np.linalg.inv(np.matmul(Xpt,Xp)),np.matmul(Xpt,Yp))
    a = a[:,0]
    
    # Create more spaced out X,Y for better plotting
    # 100 is an arbtrarilly choosen number
    X2 = np.linspace(min(X),max(X),100).tolist()
    Y2 = poly(a,X2)  
    """
    plt.plot(X,Y,'ro')
    plt.plot(X2,Y2)
    plt.show()
    """
    return a

def poly(a,x):
    result = []
    
    for term in x:
        var = 0
        for val in range(len(a)):
            var += a[val]*term**val
        result.append(var)
    
    return result  

def predict(a, x):
    var = 0
    for val in range(len(a)):
        var += a[val]*x**val
    return var


def powers(lst, start, stop):
    """Compute a matrix of powers for each number in the list."""
    # Initialize the resulting matrix
    mNew = []
    
    # Compute the powers for each number in the list
    for num in lst:
        row = [num ** i for i in range(start, stop+1)]
        mNew.append(row)
    
    return (np.array(mNew))

numpy_reg()
