# 
# CUSTOM RIDGE
#

import numpy as np

class Ridge():
    def __init__(self, iterations = 1000, learning_rate = 0.01, alpha = 1):
        self.iterations = iterations
        self.lr = learning_rate
        self.alpha = alpha

    def fit(self, X, Y):
        self.m, self.n = X.shape

        
        I = np.eye(self.n)
        self.W = np.linalg.solve(X.T @ X + self.alpha * I, X.T @ Y)

        self.b = np.zeros(Y.shape[1])
        self.X = X
        self.Y = Y

        print ("starting training tuning...")

        for i in range(self.iterations):
            Y_pred = self.predict(X)
            dY = self.Y - Y_pred
            if dY.all()== 0:
                print ("Perfect fit found at it no. ", i, "!")
                break

            dW = ( - ( 2 * ( self.X.T ) @ ( self.Y - Y_pred ) ) +               
                   ( 2 * self.alpha * self.W ) ) / self.m     
            db = - 2 * np.sum( self.Y - Y_pred ) / self.m 
            
            # update weights    
            self.W = self.W - self.lr * dW    
            self.b = self.b - self.lr * db  

        print ("Done!")

    def predict(self, X : np.ndarray) -> np.ndarray:
        return  X @ self.W + self.b

