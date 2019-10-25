import numpy as np

def cross_entropy(actuals, prediction):
    cost =   -1 * (np.dot(actuals, np.log(prediction).T) + np.dot((1 - actuals), np.log(1 - prediction).T)) / actuals.shape[1]
    # cost = - np.dot(np.log(prediction), actuals.T) / actuals.shape[1]

    return float(np.squeeze(cost))
