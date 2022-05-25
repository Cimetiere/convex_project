import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class RegularizedLinearModel():
    def __init__(self,num_predictors,tau=0.01,lambda_=0.01):
        self.tau = tau
        self.lambda_ = lambda_
        self.w = np.random.rand(num_predictors)


    def train(self,X,y, num_iterations=400):
        #self.lambda_ = 1/np.linalg.norm(X, 2)
        #print("Training started")
        for i in range(num_iterations):
            self.iterate(X,y)
        #print("Trained for {} iterations in {} minutes".format(num_iterations,12))

        if np.isnan(np.min(self.w )):
            return False
        return True

    def iterate(self,X,y):
        z = self.w - self.tau*X.transpose()@(X@self.w-y)
        self.w = np.maximum((np.abs(z)-self.tau*self.lambda_/2),0)* np.sign(z)

    def predict(self,X,y=None):
        pred = X@self.w
        if not y is None:
            rmse = np.sqrt(sum((pred-y)*(pred-y))/y.shape[0])
            return pred,rmse
        return pred, None

class LinearModel():
    def __init__(self,num_predictors,tau=0.01,lambda_=0.01):
        self.tau = tau
        self.w = np.random.rand(num_predictors)


    def train(self,X,y, num_iterations=400):
        for i in range(num_iterations):
            self.iterate(X,y)

        if np.isnan(np.min(self.w )):
            return False
        return True

    def iterate(self,X,y):
        self.w = self.w - self.tau*X.transpose()@(X@self.w-y)

    def predict(self,X,y=None):
        pred = X@self.w
        if not y is None:
            rmse = np.sqrt(sum((pred-y)*(pred-y))/y.shape[0])
            return pred,rmse
        return pred, None

def main(args={}):
    df = pd.read_csv("data.csv")
    X = df[["bathrooms", "sqft_living", "sqft_above"]]
    #X.insert(3, "bias", [1 for i in range(X.shape[0])])
    Y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, train_size=0.3)

    norm_mean = X_train.mean()
    norm_std = X_train.std()

    X_train = (X_train-norm_mean)/norm_std
    X_test = (X_test-norm_mean)/norm_std

    values_lambda =[1000,100,10,1,0.1,0.01,0.001,0.0001,0.00001]
    values_tau = [1000,100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    # values_lambda = [0.001]
    # values_tau = [0.001]

    X = X_train.to_numpy()
    y = y_train.to_numpy()

    min_rmse = np.inf
    for lambda_ in values_lambda:
        for tau in values_tau:
            model = RegularizedLinearModel(num_predictors=X.shape[1], lambda_=lambda_,tau=tau)
            has_converged = model.train(X,y)
            _, rmse = model.predict(X_test,y_test)
            print("REGU : {}:{}:{} Performance (RMSE): {}".format(lambda_, tau, has_converged, rmse))
            if rmse < min_rmse:
                min_rmse=rmse
    print("REGU best RMSE {:.2f}".format(min_rmse))

    min_rmse = np.inf
    for tau in values_tau:
        model = LinearModel(num_predictors=X.shape[1],tau=tau)
        has_converged = model.train(X,y)
        _, rmse = model.predict(X_test,y_test)
        print("DEFT : {}:{}:{} Performance (RMSE): {}".format(lambda_, tau, has_converged, rmse))
        if rmse < min_rmse:
            min_rmse=rmse
    print("DEFT best RMSE {:.2f}".format(min_rmse))


    from sklearn import linear_model
    reg = linear_model.LinearRegression(fit_intercept=False)
    reg.fit(X, y)
    y_pred = reg.predict(X_test)
    print(np.sqrt(mean_squared_error(y_test, y_pred)))
    print("The End")
if __name__ == "__main__":
    main()