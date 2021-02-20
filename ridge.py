
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
    loss = ((linearmodel(w,xTr) - yTr)**2).sum() + lambdaa * (w.transpose() @ w)
    gradient = 2 * ((linearmodel(w,xTr) - yTr) @ xTr.transpose()).sum() + 2 * lambdaa * w

    return loss,gradient
