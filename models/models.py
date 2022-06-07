import numpy as np

class RegularizedLinearModel():
    def __init__(self, num_predictors, tau=0.01, lambda_=0.01):
        self.tau = tau
        self.lambda_ = lambda_
        self.w = np.random.rand(num_predictors)

    def fit(self, X, y, num_iterations=200):
        for i in range(num_iterations):
            self.iterate(X, y)

    def iterate(self, X, y):
        z = self.w - self.tau * X.transpose() @ (X @ self.w - y)
        self.w = np.maximum((np.abs(z) - self.tau * self.lambda_ / 2), 0) * np.sign(z)

    def predict(self, X):
        pred = X @ self.w
        return pred

    def get_params(self, deep=False):
        return {'tau': self.tau, 'lambda_': self.lambda_, 'num_predictors': self.w.shape[0]}


class LinearModel():
    def __init__(self, num_predictors, tau=0.01):
        self.tau = tau
        self.w = np.random.rand(num_predictors)

    def fit(self, X, y, num_iterations=200):
        for i in range(num_iterations):
            self.iterate(X, y)

        if np.isnan(np.min(self.w)):
            return False
        return True

    def iterate(self, X, y):
        self.w = self.w - self.tau * X.transpose() @ (X @ self.w - y)

    def predict(self, X):
        pred = X @ self.w
        return pred

    def get_params(self, deep=False):
        return {'tau': self.tau, 'num_predictors': self.w.shape[0]}
