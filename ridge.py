
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
    d, n = xTr.shape
#     w = w.reshape(d,1)
    # YOUR CODE HERE
#     summ = (xTr.transpose() @ w - yTr)**2
    loss = ((xTr.transpose() @ w - yTr.reshape(n,1))**2).sum() + lambdaa * (w.transpose() @ w)
#     print(f'{xTr.shape} {yTr.shape} {w.shape}')
#     test = np.subtract(xTr.transpose() @ w, yTr.reshape(n,1))
    gradient = 2 * (xTr @ (xTr.transpose() @ w - yTr.reshape(n,1) )) + 2 * lambdaa * w
#     print(f'{loss.shape} {gradient.shape}')
    return loss,gradient

