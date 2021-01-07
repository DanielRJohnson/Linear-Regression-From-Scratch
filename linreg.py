'''
# Name: Daniel Johnson
# File: linreg.py
# Date: 1/5/2021
# Brief: This script creates uses linear regression model
#        that can be trained and create predictions
'''

import numpy as np
from costs import rmse

class LinRegModel:
    '''
    # @post: A LinRegModel object is created
    # @param: inputs: number of inputs into the network
    #         Theta: a optional parameter to set weights manually
    #         costFunc: a optional perameter to set cost function
    #                   (only one, but added for modularity)
    '''
    def __init__(self, inputs: int, Theta: np.ndarray = None, costFunc: str = "MSE") -> None:
        costFunctions = {
            "MSE": rmse
        }
        self.thetas = np.random.rand(inputs) if Theta is None else Theta
        self.cost = costFunctions[costFunc]

    '''
    # @param: X: the inputs to the model
    # @return: an np.ndarray holding the model's predictions
    '''
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.thetas)

    '''
    # @post: the model is trained by updating self.weights
    # @param: X_train: The inputs to train the model on
    #         y_train: The correct outputs from the inputs
    #         maxIters: maximum amount of training iterations
    #         alpha: learning rate, scale the length of the step in learning
    #         convergenceThreshold: the smallest number of improvement to break learning
    # @return: a list holding the cost at every training iteration
    '''
    def train(self, X_train: np.ndarray, y_train: np.ndarray, # ------>
                maxIters: int = 1000, alpha: float = 0.1, convergenceThreshold: float = 0) -> list:
        X_scale = self.scaleFeatures(X_train)
        m = y_train.shape[0]
        J_Hist = []
        J_Hist.append( self.cost(self.predict(X_scale), y_train) )
        for i in range(1, maxIters):
            hx = self.predict(X_scale)
            errors = (hx - y_train.T)
            step = (alpha * (1/m) * np.dot(errors, X_scale))
            #print("Step: ", step, "weights", self.thetas)
            self.thetas -= step[0].T

            J_Hist.append(self.cost(y_train.T, self.predict(X_scale)))
            print("Iteration: ", i, "Cost: ", J_Hist[i])

            if (np.abs(J_Hist[i - 1] - J_Hist[i]) < convergenceThreshold and i > 0):
                print("Training converged at iteration:", i)
                break
        return J_Hist

    '''
    # @param: X: the np.ndarray to be scaled
    # @return: np.ndarray of scaled inputs
    '''
    def scaleFeatures(self, X: np.ndarray) -> np.ndarray:
        X_scale = X.copy()
        for i in range(X.shape[1]):
            X_scale[:,i] = (X_scale[:,i] - X_scale[:,i].mean()) / (X_scale[:,i].std())
        return X