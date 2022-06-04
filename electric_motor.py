import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from models import RegularizedLinearModel,LinearModel
def main(args={}):
    df = pd.read_csv("cancer_reg.csv",encoding="latin-1")

    df.dropna(inplace=True)
    X = df[['avgAnnCount', 'avgDeathsPerYear', 'incidenceRate',
       'medIncome', 'popEst2015', 'povertyPercent', 'studyPerCap',
       'MedianAge', 'MedianAgeMale', 'MedianAgeFemale',
       'AvgHouseholdSize', 'PercentMarried', 'PctNoHS18_24', 'PctHS18_24',
       'PctSomeCol18_24', 'PctBachDeg18_24', 'PctHS25_Over',
       'PctBachDeg25_Over', 'PctEmployed16_Over', 'PctUnemployed16_Over',
       'PctPrivateCoverage', 'PctPrivateCoverageAlone', 'PctEmpPrivCoverage',
       'PctPublicCoverage', 'PctPublicCoverageAlone', 'PctWhite', 'PctBlack',
       'PctAsian', 'PctOtherRace', 'PctMarriedHouseholds', 'BirthRate']]
    X.fillna(X.mean())

    # for col in X:
    #     X[col+"_noise"] = X[col] + np.random.normal(loc=0, scale=X[col]*1.5, size=X[col].shape)
    #
    # for i in range(2):
    #     X['noise_{}'.format(i)] = np.random.normal(loc=np.random.randint(-5,5), scale=np.random.randint(2,5), size=X.shape[0])



    Y = df['TARGET_deathRate']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, train_size=0.8)


    norm_mean = X_train.mean()
    norm_std = X_train.std()

    X_train = (X_train-norm_mean)/norm_std
    X_test = (X_test-norm_mean)/norm_std

    X_train.insert(3, "bias", [1 for i in range(X_train.shape[0])])
    X_test.insert(3, "bias", [1 for i in range(X_test.shape[0])])


    values_lambda =[10000,1000, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    values_tau = [1000,100, 10, 1,0.05, 0.1, 0.05, 0.01,0.005, 0.001, 0.0001, 0.00001]


    X = X_train.to_numpy()
    y = y_train.to_numpy()

    num_iterations = 1000

    min_rmse = np.inf
    for lambda_ in values_lambda:
        for tau in values_tau:
            model = RegularizedLinearModel(num_predictors=X.shape[1], lambda_=lambda_,tau=tau)
            has_converged = model.train(X,y,num_iterations=num_iterations)
            _, rmse = model.predict(X_test,y_test)
            # print("REGU : {}:{}:{} Performance (RMSE): {}".format(lambda_, tau, has_converged, rmse))
            if rmse < min_rmse:
                min_rmse=rmse
    print("REGU best RMSE {:.4f}".format(min_rmse))

    min_rmse = np.inf
    for tau in values_tau:
        model = LinearModel(num_predictors=X.shape[1],tau=tau)
        has_converged = model.train(X,y,num_iterations=num_iterations)
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