import pickle
from collections import defaultdict

import nltk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import treebank
from tqdm import tqdm

from doc2vec_test import get_docs
from preprocess_text_files import get_pos_tag


def setup_most_freq_tag():
    docs = get_docs("aclImdb 2/train/pos") + get_docs("aclImdb 2/train/neg") + get_docs(
        "aclImdb 2/test/pos") + get_docs("aclImdb 2/test/neg")
    tag_freqs = defaultdict(lambda: defaultdict(int))
    for doc in tqdm(docs):
        tags = get_pos_tag(doc)
        for tok, tag in tags:
            tag_freqs[tok][tag] += 1
    with open("most_freq_tags.txt", "w+") as fp:
        for tok in tag_freqs.keys():
            max_tag, _ = max(tag_freqs[tok].items(), key=lambda x: x[1])
            fp.write("{} {}\n".format(tok, max_tag))


def plot_coefficients(classifier, feature_names, top_features=20, plot=True):
    # Taken from
    coef = classifier.coef_.toarray()[0]
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    feature_names = np.array(feature_names)
    if plot:
        plt.figure(figsize=(15, 5))
        colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
        plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
        plt.xticks(np.arange(0, 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
        plt.xlabel("Feature")
        plt.ylabel("Weighting")
        plt.axes().axvline(top_features - 0.5, color='black', linestyle='dashed')
        plt.show()
    return feature_names[top_coefficients]


# setup_most_freq_tag()

file = open("experiments_2/best/model.pickle", "rb")
model = pickle.load(file)
features = [line.strip('\n') for line in open('experiments_2/best/imdb.vocab', 'r').readlines()]
top_featuers = plot_coefficients(model, features, plot=True, top_features=15)
top_featuers = plot_coefficients(model, features, plot=False, top_features=100)

most_freq_tag = {}
pos_count = defaultdict(int)
neg_count = defaultdict(int)
for line in open("most_freq_tags.txt", "r").readlines():
    line = line.split(" ")
    most_freq_tag[line[0]] = line[1]
for i, word in enumerate(top_featuers):
    tag = most_freq_tag[word] if word in most_freq_tag else None
    if tag is None:
        tag = 'NNP'
    tag = tag.strip('\n')
    if 'NN' == tag:
        print(word)
    if i < len(top_featuers) / 2:
        neg_count[tag] += 1
    else:
        pos_count[tag] += 1

data = {'sentiment': ["Positive" if i < len(pos_count) else "Negative" for i in range(len(pos_count) + len(neg_count))],
        "tag": list(pos_count.keys()) + list(neg_count.keys()),
        "count": list(pos_count.values()) + list(neg_count.values())}
df = pd.DataFrame.from_dict(data)

g = sns.catplot(x="tag", y="count", hue="sentiment", data=df, kind="bar")

g.set_axis_labels("POS tag", "Count")
plt.show()

class_probs = {
    "1": 0.9358821186778176,
    "2": 0.9079061685490878,
    "3": 0.8689492325855962,
    "4": 0.7741935483870968,
    "5": 0,
    "6": 0,
    "7": 0.7923710446467274,
    "8": 0.8743859649122807,
    "9": 0.8946245733788396,
    "10": 0.9105821164232847
}
plt.bar(class_probs.keys(), class_probs.values())
plt.ylim(bottom=0.7)
plt.xlabel("Review score")
plt.ylabel("Accuracy")
plt.show()
