from collections import defaultdict

from libsvm.svmutil import svm_read_problem, svm_train, svm_save_model, svm_predict, svm_load_model

train = True
test = False

m = None
if train:
    y, x = svm_read_problem('/Users/james_hargreaves/PycharmProjects/r250/aclImdb 2/train/labeledBow.feat')
    y = [1 if a > 5 else -1 for a in y]
    y = y[:20000]
    x = x[:20000]
    m = svm_train(y, x, '-t 0')
    svm_save_model("models/model_test.model",m)
if test:
    if m is None:
        m = svm_load_model("models/model_test.model")
    true_y, x = svm_read_problem('/Users/james_hargreaves/PycharmProjects/r250/aclImdb 2/test/labeledBow.feat')
    true_y_classes = [1 if a > 5 else -1 for a in true_y]
    result = svm_predict(true_y_classes, x, m)
    accuracy = result[1][0]
    pred_y = result[0]
    count = 0
    total_counts = defaultdict(int)
    total_correct = defaultdict(int)
    for pred,true in zip(pred_y,true_y):
        true_class = 1 if true > 5 else -1
        if true_class * pred > 0:
            count += 1
            total_correct[int(true)] += 1
        total_counts[int(true)] += 1
    print("Accuracy = ", count/len(pred_y))
    for key in total_counts:
        print("accuracy of class {} = {}".format(key,total_correct[key]/total_counts[key]))
    x = 0


