import matplotlib

import numpy as np
import sklearn
from scipy import sparse
from sklearn.datasets import fetch_openml
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
# From https://thuijskens.github.io/2016/12/29/bayesian-optimisation/
from bayes_opt.gp import bayesian_optimisation
from bayes_opt.plotters import plot_iteration

default_coef0 = 0
default_degree = 3
X_train, y_train = None, None


def setup(feature_path, kernel_name, multiclass=False):
    global X_train, y_train, kernel
    X_train, y_train = sklearn.datasets.load_svmlight_file(feature_path)
    if not multiclass:
        y_train = [1 if a > 5 else -1 for a in y_train]
    kernel = kernel_name


def sample_loss(params):
    global X_train, y_train
    print(params)
    C = params[0]
    gamma = 0
    degree = default_degree
    coef0 = default_coef0
    if len(params) >= 2:
        gamma = params[1]
    if len(params) >= 3:
        coef0 = params[2]
    if len(params) >= 4:
        degree = int(params[3])

    model = SVC(C=10 ** C, gamma=10 ** gamma, degree=degree, coef0=coef0, kernel=kernel, random_state=12345)

    xs = sparse.vstack((X_train[:1000], X_train[24000:]))
    ys = np.array(y_train[:1000] + y_train[24000:])
    return cross_val_score(model,
                           X=xs,
                           y=ys,
                           scoring='roc_auc',
                           cv=3).mean()


def get_optimum_hyperparams(kernel, limits, should_plot, iterations=40):
    bounds = np.array([v for v in limits.values()])
    xp, yp = bayesian_optimisation(iterations, sample_loss, bounds)
    xp = np.array(xp)
    yp = np.array(yp)
    max_val = yp.max()
    max_point = xp[yp.argmax(), :]
    if should_plot:
        lambdas = np.linspace(limits["c"][0], limits["c"][1], 25)
        if "gamma" in limits:
            gammas = np.linspace(limits["gamma"][0], limits["gamma"][1], 20)
        else:
            gammas = None
        matplotlib.rc('text', usetex=False)
        plot_iteration(lambdas, xp, yp, first_iter=3, second_param_grid=gammas, optimum=max_point,
                       filepath="results_bayes_optimisation/v2/{}".format(kernel))
        plt.show()
    return max_point, max_val

if __name__ == "__main__":
    X_train, y_train = sklearn.datasets.load_svmlight_file('aclImdb 2/train/labeledBowNoPos.feat')
    y_train = [1 if a > 5 else -1 for a in y_train]

    # kernel = 'linear'
    # lims = {"c": [-5, 7]}
    # print(kernel)
    # print(get_optimum_hyperparams(kernel, lims, True, iterations=40))
    # [-2.05956204] => 0.8928927488924613

    # kernel = 'poly'
    # lims = {"c": [-5, 7], "gamma": [-5, 7], "coef0": [0, 5], "degree": [2, 5]}
    # print(kernel)
    # print(get_optimum_hyperparams(kernel, lims, False, iterations=20))
    # (array([ 0.4359923 , -4.19174131,  4.105554  ,  4.41044495]), 0.8809290580253158)

    # kernel = 'poly'
    # default_coef0 = 4.105554
    # default_degree = 3
    # lims = {"c": [-5, 10], "gamma": [-7, 5]}
    # print(kernel)
    # print(get_optimum_hyperparams(kernel, lims, True, iterations=40))


    # kernel = 'sigmoid'
    # lims = {"c": [-5, 7], "gamma": [-5, 7], "coef0": [-5, 15]}
    # print(kernel)
    # print(get_optimum_hyperparams(kernel, lims, False, iterations=20))
    # (array([ 3.56481912, -2.2307039 , 13.02698483]), 0.6628290866616018)

    kernel = 'rbf'
    lims = {"c": [-5, 7], "gamma": [-5, 7]}
    print(kernel)
    print(get_optimum_hyperparams(kernel, lims, True, iterations=40))
    # (array([ 3.23949514, -4.59848105]), 0.8767388074629449)

    # Run with coef0 value
    # kernel = 'sigmoid'
    # lims = {"c": [-5, 7], "gamma": [-5, 7]}
    # default_coef0 = 13.02698483
    # print(kernel, default_coef0)
    # print(get_optimum_hyperparams(kernel, lims, True, iterations=40))
    # (array([-3.03028767, -2.71955719]), 0.6824522594918739)

