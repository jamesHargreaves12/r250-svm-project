import os
import pickle
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.datasets
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

import bayes_opt_attempt

train_feats_path = 'experiments_2/for_analysis/train/labeledBow.feat'
test_feats_path = 'experiments_2/for_analysis/test/labeledBow.feat'
X_train, y_train = sklearn.datasets.load_svmlight_file(train_feats_path)
X_test, y_test = sklearn.datasets.load_svmlight_file(test_feats_path, n_features=X_train.shape[1])
max_val = -1
for kernel in ['linear', 'rbf', 'sigmoid', 'poly']:
    print("Testing {} kernel".format(kernel))
    if kernel == 'linear':
        lims = {"c": [-5, 7]}
    else:
        lims= {"c": [-5, 7], "gamma": [-5, 7]}
    print("Setting up Training Params")
    bayes_opt_attempt.setup(train_feats_path, kernel)
    print("Optimising")
    point, val = bayes_opt_attempt.get_optimum_hyperparams(kernel, lims, False, iterations=20)
    print("{} at {} => {}".format(kernel, point, val))
    if val > max_val:
        max_point = point
        max_kernel = kernel
        max_val = val
print("Best kernel = {}".format(max_kernel))
print(max_point, max_kernel)

kernel = max_kernel
C = float(max_point[0])
model_path = "models/multiclass_test_{}_new.model".format(C)
# This currently doesnt handdle the gamma case

model = SVC(C=10 ** C, kernel=kernel, random_state=12345)
if os.path.exists(model_path):
    model = pickle.load(open(model_path, "rb"))
else:
    model.fit(X_train, y_train)
    file = open(model_path, "wb+")
    pickle.dump(model, file)
print("Predicting")
pred_y = model.predict(X_test)
acc = 0
for pred, test in zip(y_test, pred_y):
    pred_val = 1 if pred > 5 else -1
    test_val = 1 if test > 5 else -1
    if pred_val * test_val > 0:
        acc += 1
print("Overall Acc ", acc / len(y_test))
cm = confusion_matrix(y_test, pred_y)
print(cm)
# cm = [[4010, 209, 206, 235, 38, 54, 11, 259],
#       [1324, 183, 244, 313, 45, 36, 6, 151],
#       [996, 188, 376, 588, 126, 74, 13, 180],
#       [712, 154, 382, 803, 185, 142, 28, 229],
#       [131, 38, 105, 304, 561, 476, 83, 609],
#       [136, 16, 82, 194, 461, 589, 136, 1236],
#       [100, 12, 43, 98, 204, 360, 126, 1401],
#       [246, 23, 57, 106, 196, 434, 171, 3766]]
df_cm = pd.DataFrame(cm, index=[1, 2, 3, 4, 7, 8, 9, 10],
                     columns=[1, 2, 3, 4, 7, 8, 9, 10])
plt.figure(figsize=(10, 7))
g = sn.heatmap(df_cm, annot=True, fmt='d')
g.set(xlabel="Actual Class", ylabel="Predicted Class")
plt.show()
