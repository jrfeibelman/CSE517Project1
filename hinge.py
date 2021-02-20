from numpy import maximum
import numpy as np


def hinge(w,xTr,yTr,lambdaa):
#
#
# INPUT:
# xTr dxn matrix (each column is an input vector)
# yTr 1xn matrix (each entry is a label)
# lambda: regularization constant
# w weight vector (default w=0)
#
# OUTPUTS:
#
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w


    # YOUR CODE HERE
    
    
#     print(f'{xTr.shape} {yTr.shape} {w.shape} {xx.shape}')
    comparator = 1 - yTr @ (xTr.transpose().dot(w))
    loss = maximum(0, comparator).sum() + lambdaa * (w.transpose() @ w)
    
    if comparator > 0:
        gradient = 2 * lambdaa * w - (xTr @ yTr.transpose())
    else:
        gradient = 2 * lambdaa * w

    return loss,gradient
