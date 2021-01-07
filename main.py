'''
# Name: Daniel Johnson
# File: main.py
# Date: 1/5/2021
# Brief: This script trains a linreg model and plots its predictions
'''

import numpy as np
import matplotlib.pyplot as plt
from linreg import LinRegModel
import pandas as pd

def main():
    #get the data
    data = pd.read_csv('data/kc_house_data_NaN.csv')

    #get the columns for sqft, beds, and price
    X = data.iloc[:,np.r_[4,6]].to_numpy()
    y = data.iloc[:,3].to_numpy()

    #make the model
    lr = LinRegModel(X.shape[1])

    #train it
    J_Hist = lr.train(X, y, maxIters = 1 * (10**3), alpha = 1 * (10**-5), convergenceThreshold = 1 * (10**-5) )

    #draw it
    draw(lr, X, y, J_Hist, bool3D=True)

'''
# @post: graphs are shown on the screen from the model, inputs, outputs, and cost history
'''
def draw(model, X: np.ndarray, y: np.ndarray, J_Hist: list, bool3D: bool = False):
    #set up subplots for the cost history and prediction graph
    fig = plt.figure()
    fig.suptitle('Linear Regression') #supertitle
    fig.tight_layout(pad=2.5, w_pad=1.5, h_pad=0) #fix margins

    costPlot = fig.add_subplot(121)
    drawCostHistory(J_Hist, costPlot)

    if bool3D:
        predPlot = fig.add_subplot(122, projection='3d')
        drawPrediction3D(model, X, y, predPlot)
    else:
        predPlot = fig.add_subplot(122)
        drawPrediction2D(model, X, y, predPlot)
    #show the cool graphs :)
    plt.show()

'''
# @post: cost history is plotted to the screen
'''
def drawCostHistory(J_Hist: list, plot) -> None:
    plot.plot(J_Hist)
    plot.set_ylabel('Cost')
    plot.set_xlabel('Iterations')
    plot.set_title('Cost vs. Iterations')
    plot.axis([0, len(J_Hist), 0, max(J_Hist)])
    plot.set_aspect(len(J_Hist)/max(J_Hist))

'''
# @post: predictions are plotted to the screen in 2D
'''
def drawPrediction2D(model, X: np.ndarray, y: np.ndarray, plot) -> None:
    plot.scatter(X, y)
    plot.plot(X, model.predict(X))
    plot.set(xlabel='X', ylabel='Y')
    plot.set_title('Prediction')
    plot.set_aspect(max(X)/max(y))

'''
# @post: predictions are plotted to the screen in 3D
'''
def drawPrediction3D(model, X: np.ndarray, y: np.ndarray, plot) -> None:
    plot.scatter(X[:,0], X[:,1], y, s=0.5, c="blue")
    plot.scatter(X[:,0], X[:,1], model.predict(X), s=5, c="red")
    plot.set(xlabel='X', ylabel='Y', zlabel='Z')
    plot.set_title('Prediction\nBlue = Real, Red = Predicted')

if __name__ == "__main__": main()