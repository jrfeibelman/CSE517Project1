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
        
        if stepsize < eps:
            stepsize = eps
        
        dW = -1 * stepsize * gradient
        
        w = w + dW
        
        nextLoss, nextGradient = func(w)
                
        if nextLoss < loss:
            stepsize = stepsize * 1.01
            loss = nextLoss
            gradient = nextGradient
        else:
            stepsize = stepsize * 0.5
            w = w - dW
        
        print(loss)

    return w