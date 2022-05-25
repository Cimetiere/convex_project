import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from models import RegularizedLinearModel,LinearModel
def main(args={}):
    df = pd.read_csv("winequality-red.csv")


    X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]

    Y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, train_size=0.8)


    norm_mean = X_train.mean()
    norm_std = X_train.std()

    X_train = (X_train-norm_mean)/norm_std
    X_test = (X_test-norm_mean)/norm_std

    X_train.insert(3, "bias", [1 for i in range(X_train.shape[0])])
    X_test.insert(3, "bias", [1 for i in range(X_test.shape[0])])


    values_lambda =[1000,100,10,1,0.1,0.01,0.001,0.0001,0.00001]
    values_tau = [1000,100, 10, 1,0.05, 0.1, 0.05, 0.01,0.005, 0.001, 0.0001, 0.00001]


    X = X_train.to_numpy()
    y = y_train.to_numpy()

    min_rmse = np.inf
    for lambda_ in values_lambda:
        for tau in values_tau:
            model = RegularizedLinearModel(num_predictors=X.shape[1], lambda_=lambda_,tau=tau)
            has_converged = model.train(X,y)
            _, rmse = model.predict(X_test,y_test)
            # print("REGU : {}:{}:{} Performance (RMSE): {}".format(lambda_, tau, has_converged, rmse))
            if rmse < min_rmse:
                min_rmse=rmse
    print("REGU best RMSE {:.4f}".format(min_rmse))

    min_rmse = np.inf
    for tau in values_tau:
        model = LinearModel(num_predictors=X.shape[1],tau=tau)
        has_converged = model.train(X,y)
        _, rmse = model.predict(X_test,y_test)
        # print("DEFT : {}:{}:{} Performance (RMSE): {}".format(lambda_, tau, has_converged, rmse))
        if rmse < min_rmse:
            min_rmse=rmse
    print("DEFT best RMSE {:.4f}".format(min_rmse))


    from sklearn import linear_model
    reg = linear_model.LinearRegression(fit_intercept=True)
    reg.fit(X, y)
    y_pred = reg.predict(X_test)
    print(np.sqrt(mean_squared_error(y_test, y_pred)))
    print("The End")
if __name__ == "__main__":
    main()