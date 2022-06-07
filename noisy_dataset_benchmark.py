import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from models import RegularizedLinearModel,LinearModel
from sklearn.model_selection import cross_val_score
def main():

    df = pd.read_csv("data/cancer_reg.csv", encoding="latin-1")

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

    for col in X:
        X[col+"_noise"] = X[col] + np.random.normal(loc=0, scale=X[col]*1.5, size=X[col].shape)

    for i in range(2):
        X['noise_{}'.format(i)] = np.random.normal(loc=np.random.randint(-5,5), scale=np.random.randint(2,5), size=X.shape[0])

    Y = df['TARGET_deathRate']

    norm_mean = X.mean()
    norm_std = X.std()

    X = (X-norm_mean)/norm_std

    X.insert(3, "bias", [1 for i in range(X.shape[0])])

    values_lambda =[10000,1000, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    values_tau = [1000,100, 10, 1,0.05, 0.1, 0.05, 0.01,0.005, 0.001, 0.0001, 0.00001]

    X = X.to_numpy()
    y = Y.to_numpy()
    num_cv = 5
    mse_grid_search_RPGD = []
    for lambda_ in values_lambda:
        for tau in values_tau:
            model = RegularizedLinearModel(num_predictors=X.shape[1], lambda_=lambda_,tau=tau)
            res = cross_val_score(model,
                                X,
                                y,
                                cv=num_cv,
                                scoring='neg_mean_squared_error')
            mse_grid_search_RPGD.append(sum(res)/len(res))

    mse_grid_search_GD = []
    for tau in values_tau:
        model = LinearModel(num_predictors=X.shape[1],tau=tau)
        res = cross_val_score(model,
                        X,
                        y,
                        cv=num_cv,
                        scoring='neg_mean_squared_error')
        mse_grid_search_GD.append(sum(res) / len(res))



    from sklearn import linear_model
    reg = linear_model.LinearRegression(fit_intercept=False)
    res = cross_val_score(reg,
                    X,
                    y,
                    cv=num_cv,
                    scoring='neg_mean_squared_error')

    print()
    print("Scikit-learn Least Square:                  {:.2f}".format(-sum(res) / len(res)))
    print("Mean Square Error Vanilla Least Square:     {:.2f}".format(-max(list(filter(lambda x: not np.isnan(x),mse_grid_search_GD)))))
    print("Mean Square Error regularized Least Square: {:.2f}".format(-max(list(filter(lambda x: not np.isnan(x),mse_grid_search_RPGD)))))

if __name__ == "__main__":
    main()