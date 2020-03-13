import json
import os
import pickle
import re
import string
from random import random

import nltk
from nltk import word_tokenize
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

NOUN_TAGS = ["NN", "NNP", "NNS", "NNPS"]
ADJ_TAGS = ["JJ", "JJR", "JJS"]
ADV_TAGS = ["RB", "RBR", "RBS"]

remove = string.punctuation
remove = remove.replace("-", "")
pattern = re.compile("[{}]".format(remove))

remove_non_sentiment = remove.replace("!", "").replace("?", "")
pattern_non_sentiment = re.compile("[{}]".format(remove_non_sentiment))
pattern_multi_excalm = re.compile("!!+")
pattern_multi_question = re.compile("\?\?+")

pos_cache_file = "data_files/pos_cache.pickle"
pos_cache = pickle.load(open(pos_cache_file, 'rb')) if os.path.exists(pos_cache_file) else {}


def get_pos_tag(text):
    if text in pos_cache:
        return pos_cache[text]
    else:
        tokens = word_tokenize(text)
        tags = nltk.pos_tag(tokens)
        pos_cache[text] = tags
        if random() < 0.001:
            with open(pos_cache_file, 'wb') as f:
                pickle.dump(pos_cache, f, pickle.HIGHEST_PROTOCOL)
        return tags


def preprocess(text, config):
    if "remove_punc" in config and config["remove_punc"]:
        text = pattern.sub(' ', text)
    elif "remove_non_sentiment_punc" in config and config["remove_non_sentiment_punc"]:
        text = pattern_non_sentiment.sub(' ', text)
        text = pattern_multi_excalm.sub(' !! ', text)
        text = pattern_multi_question.sub(' ?? ', text)

    pos_tagged = get_pos_tag(text)
    tokens = [(w.lower(), t) for w, t in pos_tagged if w.lower() not in ENGLISH_STOP_WORDS]

    if "only_alphanumeric" in config and config["only_alphanumeric"]:
        tokens = [(w, t) for w, t in pos_tagged if w.isalpha()]

    if "remove_nouns" in config and config["remove_nouns"]:
        tokens = [(w, t) for w, t in pos_tagged if t not in NOUN_TAGS]

    if "only_adverbs_adjectives" in config and config["only_adverbs_adjectives"]:
        tokens = [(w, t) for w, t in pos_tagged if t in ADJ_TAGS + ADV_TAGS]

    if "with_pos_tag" in config and config["with_pos_tag"]:
        return " ".join(["{}_{}".format(w, t) for w, t in tokens])
    else:
        return " ".join(["{}".format(w) for w, t in tokens])


def run_preprocess(in_path, config):
    if "preprocess_outpath_format" in config:
        parts = in_path.split("/")
        out_path = config["preprocess_outpath_format"].format(parts[-2], parts[-1])
    else:
        out_path = os.path.join(os.path.dirname(in_path), "{}_preprocess".format(os.path.basename(in_path)))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    in_list = os.listdir(in_path)
    for file_path in in_list:
        with open(os.path.join(in_path, file_path), "r") as in_file:
            with open(os.path.join(out_path, file_path), "w+") as out_file:
                text = in_file.read()
                result = preprocess(text, config)
                out_file.write(result)


def run_preprocess_v2(in_path, out_path, config):
    parent_loc = os.path.abspath(os.path.join(out_path, os.pardir))
    if not os.path.exists(parent_loc):
        os.makedirs(parent_loc)
    in_list = os.listdir(in_path)
    results = []
    for file_path in in_list:
        with open(os.path.join(in_path, file_path), "r") as in_file:
            text = in_file.read()
            result = preprocess(text, config)
            results.append({"file": file_path, "text": result})
    with open(out_path, "w+") as out_file:
        out_file.write('\n'.join([json.dumps(r) for r in results]))


if __name__ == "__main__":
    conf = {"with_pos_tag": True}

    run_preprocess('aclImdb 2/test/neg', conf)
    run_preprocess('aclImdb 2/test/pos', conf)
    run_preprocess('aclImdb 2/train/neg', conf)
    run_preprocess('aclImdb 2/train/pos', conf)
