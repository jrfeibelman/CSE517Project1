
import numpy as np


def ridge(w,xTr,yTr,lambdaa):
#
# INPUT:
# w weight vector (default w=0)
# xTr:dxn matrix (each column is an input vector)
# yTr:1xn matrix (each entry is a label)
# lambdaa: regression constant
#
# OUTPUTS:
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
#
# [d,n]=size(xTr);

    # YOUR CODE HERE
#     summ = (xTr.transpose() @ w - yTr)**2
#     print(f'{xTr.shape} {yTr.shape} {w.shape} {summ.shape}')
    loss = ((xTr.transpose() @ w - yTr)**2).sum() + lambdaa * (w.transpose() @ w)
    
    gradient = 2 * ((xTr.transpose() @ w - yTr) @ xTr.transpose()).sum() + 2 * lambdaa * w

    return loss,gradient
