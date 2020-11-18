from sklearn.linear_model import Lasso
import pandas as pd
import utility
import config
import optuna

def run(trial):
    # read the training data with fold
    df = pd.read_csv(config.Training_file+'reg_stratified.csv')
    df.drop(
        ['Unnamed: 0', 'GT Compressor inlet air pressure (P1) [bar]', 'GT Compressor inlet air temperature (T1) [C]'],
        axis=1, inplace=True)

    # training data is where data is not equal fold
    df_train = df[df.kfold != 4].reset_index(drop=True)
    # validation data is where data is equal fold
    df_valid = df[df.kfold == 4].reset_index(drop=True)

    feature = df.drop(['kfold', 'GT Compressor decay state coefficient.'], axis=1).columns

    x_train = df_train[feature].values
    y_train = df_train['GT Compressor decay state coefficient.'].values

    x_valid = df_valid[feature].values
    y_valid = df_valid['GT Compressor decay state coefficient.'].values

    alpha = trial.suggest_uniform('alpha', 0.1, 1)
    max_iter = trial.suggest_int('max_iter', 1000, 10000)
    selection = trial.suggest_categorical('selection', ['cyclic', 'random'])

    clf = Lasso(alpha=alpha, fit_intercept=True, normalize=False, precompute=False, max_iter=max_iter,
                tol=0.0001, warm_start=False, positive=False, random_state=None, selection=selection)

    clf.fit(x_train, y_train)

    pred = clf.predict(x_valid)

    accuracy = utility.root_mean_squared(y_valid, pred)
    print(f" Accuracy = {accuracy} ")

    return accuracy


if __name__ == "__main__" :
    study = optuna.create_study(direction='minimize')
    study.optimize(run, n_trials=100)

    trial = study.best_trial

    print('Accuracy {}'.format(trial.value))
    print('Best Hyperparameters {}'.format(trial.params))
