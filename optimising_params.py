import os

from libsvm.commonutil import svm_read_problem
from libsvm.svmutil import svm_train, svm_save_model, svm_load_model, svm_predict
from bayes_opt import BayesianOptimization

# For now just consider linear
y, x = svm_read_problem('/Users/james_hargreaves/PycharmProjects/r250/aclImdb 2/train/labeledBow.feat')
y = [1 if a > 5 else -1 for a in y]
train_x, valid_x = x[:11500] + x[12500:24000], x[11500:12500] + x[24000:]
train_y, valid_y = y[:11500] + y[12500:24000], y[11500:12500] + y[24000:]


def get_similar_test(c):
    for file_name in os.listdir("models"):
        if file_name.startswith("model_lin_{}".format(c)):
            return os.path.join("models",file_name)
    return None


def get_score_lin(c):
    model_path = "models/model_lin_{}.model".format(c)
    similar_file = get_similar_test(c)
    if similar_file:
        m = svm_load_model(similar_file)
    else:
        m = svm_train(train_y, train_x, '-t 0 -c {}'.format(c))
        svm_save_model(model_path, m)
    result = svm_predict(valid_y, valid_x, m)
    accuracy = result[1][0]
    return accuracy


def get_score_rbf(c, gamma):
    model_path = "models/model_rbf_{}_{}.model".format(c, gamma)
    if os.path.isfile(model_path):
        m = svm_load_model(model_path)
    else:
        m = svm_train(train_y, train_x, '-t 2 -c {} -g {}'.format(c, gamma))
        svm_save_model(model_path, m)
    result = svm_predict(valid_y, valid_x, m)
    accuracy = result[1][0]
    return accuracy


# get_score_lin(pow(10, -30), 1, 1)
# get_score_lin(1,0.5,0.1)

pbounds = {'c': (pow(10, -2), pow(10, 5))}
# pbounds = {'c': (pow(10, -2), pow(10, 5)), 'gamma': (pow(10,-3), pow(10,3))}
optimizer = BayesianOptimization(
    f=get_score_lin,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)
optimizer.maximize()
print(optimizer.max)

