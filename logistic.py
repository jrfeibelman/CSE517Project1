import math
import numpy as np

'''

    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0)

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''
def logistic(w,xTr,yTr):
    
#     print(f'{xTr.shape} {yTr.shape} {w.shape} {w.shape}')

    loss = np.log(1+np.exp(-1 * yTr @ (xTr.transpose() @ w))).sum()
    numerator = (np.dot(xTr,yTr.transpose()) @ np.exp(-1 * yTr @ (xTr.transpose() @ w)))
    denom = (1+np.exp(-1 * yTr @ (xTr.transpose() @ w)))
    gradient = -1*((numerator)/(denom))
    
#     print(f'Num:{numerator.shape}, Denom:{denom.shape}, Grad:{gradient.shape} {w.shape}')
#     print(f'Loss: {loss}, Gradient: {gradient}')

    return loss,gradient
