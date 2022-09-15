from os import stat_result
from tracemalloc import Statistic
import numpy as np
import matplotlib.pyplot as plt
import math


class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = np.array(theta).reshape(-1, 1)
        self.bounds = None

    @staticmethod
    def add_intercept(x):
        if (type(x) != np.ndarray or len(x) == 0
        or len(x.shape) != 2):
            return None
        try:
            return np.insert(x, 0, 1, axis=1).astype(float)
        except Exception as e:
            print("Error in add_intercept", e)
            return None

    @staticmethod
    def sigmoid_(x):
        if type(x) is not np.ndarray:
            print("TypeError in sigmoid")
            return None
        try:
            return np.array(1 / (1 + np.exp(-x))).reshape(-1, 1)
        except Exception as e:
            print("Error in sigmoid", e)
            return None

    @staticmethod
    def loss_element_(y, y_hat, eps=1e-15):
        if (type(y) is not np.ndarray or type(y_hat) is not np.ndarray
           or len(y.shape) != 2 or len(y_hat.shape) != 2
           or y.shape[1] != 1 or y_hat.shape[1] != 1
           or y.shape[0] != y_hat.shape[0]):
            print("TypeError in log_loss")
            return None
        try:
            return -(y * np.log(y_hat + eps)
                            + (1 - y) * np.log(1 - y_hat + eps))
        except Exception as e:
            print("Error in log_loss", e)
            return None

    def loss_(self, y, y_hat, eps=1e-15):
        if (type(y) is not np.ndarray or type(y_hat) is not np.ndarray
           or len(y.shape) != 2 or len(y_hat.shape) != 2
           or y.shape[1] != 1 or y_hat.shape[1] != 1
           or y.shape[0] != y_hat.shape[0]):
            print("TypeError in log_loss")
            return None
        try:
            # print(self.loss_element_(y, y_hat, eps))
            return np.sum(self.loss_element_(y, y_hat, eps)) / y.shape[0]
        except Exception as e:
            print("Error in log_loss", e)
            return None

    def predict_(self, x):
        if (type(x) is not np.ndarray or type(self.theta) is not np.ndarray
        or len(x.shape) != 2 or len(self.theta.shape) != 2
        or x.shape[1] + 1 != self.theta.shape[0] or self.theta.shape[1] != 1):
            print("TypeError in predict")
            return None
        try:
            return self.sigmoid_(np.dot(self.add_intercept(x), self.theta))
        except Exception as e:
            print("Error in predict", e)
            return None

    def gradient_(self, x, y):
        if (type(x) is not np.ndarray or type(y) is not np.ndarray
        or type(self.theta) is not np.ndarray or x.size == 0 or y.size == 0
        or self.theta.size == 0 or x.shape[1] + 1 != self.theta.shape[0]
        or x.shape[0] != y.shape[0] or y.shape[1] != 1 or self.theta.shape[1] != 1):
            print("TypeError in gradient")
            return None
        try:
            return np.dot(self.add_intercept(x).T, self.predict_(x) - y)  / x.shape[0]
        except Exception as e:
            print("Error in gradient", e)
            return None

    def fit_(self, x, y):
        if (type(x) is not np.ndarray or type(y) is not np.ndarray
        or x.size == 0 or y.size == 0 or x.shape[1] + 1 != self.theta.shape[0]
        or x.shape[0] != y.shape[0] or y.shape[1] != 1 or self.theta.shape[1] != 1):
            print("TypeError in fit")
            return None
        try:
            for i in range(self.max_iter):
                try:
                    self.theta -= self.alpha * self.gradient_(x, y)
                    if (math.isnan(self.theta[0])):
                        return self.theta
                except RuntimeWarning:
                    self.theta[0] = math.nan
                    return self.theta
        except Exception as e:
            print("Error in fit", e)
            return None

    def minmax_(self, data):
        if (type(data) != np.ndarray or len(data) == 0):
            print("TypeError in minmax")
            return None
        try:
            if self.bounds is None:
                self.bounds = np.array([data.min(), data.max()])
            print(self.bounds)
            return (data - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        except Exception as e:
            print("Error in minmax: ", e)
            return None

    def reverse_minmax_(self, data):
        return data * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

    def plot_(self, x, y, xlabel="x", ylabel="y", units="units"):
        plt.plot(x, y, 'o',
                 label="$s_{true}(" + units + ")$",
                 color="deepskyblue")
        plt.plot(x, self.predict_(x), '+',
                 color='limegreen',
                 label="$s_{predict}(" + units + ")$")
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.title("Cost: " + str(self.mse_(y, self.predict_(x))))
        plt.grid()
        plt.show()