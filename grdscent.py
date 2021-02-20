import numpy as np

def grdescent(func,w0,stepsize,maxiter,tolerance=1e-02):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
    eps = 2.2204e-14 #minimum step size for gradient descent
 
    # YOUR CODE HERE
    
    w = w0
    loss, gradient = func(w0)
    for n in range(maxiter):        
        if np.linalg.norm(gradient) < tolerance:
            return w
        
        if stepSize < eps:
            stepSize = eps
        
        deltaW = -1 * stepsize * gradient
        
        w = w + deltaW
        
        nextLoss, gradient = func(w)
                
        if nextLoss < loss:
            stepSize *= 1.01
        else:
            stepSize *= 0.5
        
        loss = nextLoss

    return w