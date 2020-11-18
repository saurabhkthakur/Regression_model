import joblib
import pandas as pd
import utility
import os
import config
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error as mse
import numpy as np

def run(fold):
    # read the training data with fold
    df = pd.read_csv(config.Training_file+'reg_stratified.csv')
    df.drop(
        ['Unnamed: 0', 'GT Compressor inlet air pressure (P1) [bar]', 'GT Compressor inlet air temperature (T1) [C]'],
        axis=1, inplace=True)
    df_test = df[df.kfold == 4].reset_index(drop=True)

    # training data is where data is not equal fold
    df_train = df[(df.kfold != fold) & (df.kfold != 4)].reset_index(drop=True)
    # validation data is where data is equal fold
    df_valid = df[(df.kfold == fold) & (df.kfold != 4)].reset_index(drop=True)

    feature = df.drop(['kfold', 'GT Turbine decay state coefficient.'], axis=1).columns

    test_feat = df.drop(['kfold', 'GT Compressor decay state coefficient.'], axis=1).columns

    X_test = df_test[test_feat].values
    y_test = df_test['GT Compressor decay state coefficient.'].values

    x_train = df_train[feature].values
    y_train = df_train['GT Turbine decay state coefficient.'].values

    x_valid = df_valid[feature].values
    y_valid = df_valid['GT Turbine decay state coefficient.'].values

    clf = Lasso(alpha=0.10211892570954104, fit_intercept=True, normalize=False, precompute=False, max_iter=5717,
                tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')

    clf.fit(x_train, y_train)

    pred = clf.predict(x_valid)

    accuracy = np.sqrt(mse(y_valid, pred))

    test_pred = clf.predict(X_test)

    print(test_pred, y_test)
    print(f"Fold={fold}, Accuracy = {accuracy} ")


    print(np.sqrt(mse(y_test, test_pred)))
    joblib.dump(
        clf,
        f'{config.Model_output}+dt_{fold}.bin'
    )

    return accuracy


if __name__ == "__main__" :

    count = 0
    for i in range(4):
        count += run(i)

    print("The accuracy is ",count / 4)








