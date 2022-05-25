import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class RegularizedLinearModel():
    def __init__(self, num_predictors, tau=0.01, lambda_=0.01):
        self.tau = tau
        self.lambda_ = lambda_
        self.w = np.random.rand(num_predictors)

    def train(self, X, y, num_iterations=100):
        # self.lambda_ = 1/np.linalg.norm(X, 2)
        # print("Training started")
        for i in range(num_iterations):
            self.iterate(X, y)

        # print("Trained for {} iterations in {} minutes".format(num_iterations,12))

        if np.isnan(np.min(self.w)):
            return False
        return True

    def iterate(self, X, y):
        z = self.w - self.tau * X.transpose() @ (X @ self.w - y)
        self.w = np.maximum((np.abs(z) - self.tau * self.lambda_ / 2), 0) * np.sign(z)

    def predict(self, X, y=None):
        pred = X @ self.w
        if not y is None:
            rmse = sum((pred - y) * (pred - y)) / y.shape[0]
            return pred, rmse
        return pred, None


class LinearModel():
    def __init__(self, num_predictors, tau=0.01, lambda_=0.01):
        self.tau = tau
        self.w = np.random.rand(num_predictors)

    def train(self, X, y, num_iterations=100):
        for i in range(num_iterations):
            self.iterate(X, y)

        if np.isnan(np.min(self.w)):
            return False
        return True

    def iterate(self, X, y):
        self.w = self.w - self.tau * X.transpose() @ (X @ self.w - y)

    def predict(self, X, y=None):
        pred = X @ self.w
        if not y is None:
            rmse =sum((pred - y) * (pred - y)) / y.shape[0]
            return pred, rmse
