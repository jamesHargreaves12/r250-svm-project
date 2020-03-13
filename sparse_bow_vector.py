import json
import logging
import os
import re
import string
import numpy as np

import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, strip_accents_ascii
from wordfreq import word_frequency

re_pattern_tok = '(?u)\S+'


def get_text_from_files(in_path):
    result = []
    for line in open(in_path).readlines():
        value = json.loads(line.strip('\n'))
        result.append((value["file"], value["text"]))
    return result


def remove_words_present_in_one_doc(locations, vocab_filepath, config):
    text_files = [get_text_from_files(l) for l in locations]
    texts = []
    for files in text_files:
        texts.extend([strip_accents_ascii(x) for _, x in files])
    vocab_file = open(vocab_filepath, "r")
    vocab = set([line.strip('\n') for line in vocab_file.readlines()])
    seen_once = set()
    seen_atleast_twice = set()
    tok_re = re.compile(re_pattern_tok)
    for text in texts:
        toks = tok_re.findall(text)
        toks = [t.lower() for t in toks]
        for t in set(toks):
            if t in seen_atleast_twice:
                continue
            elif t in seen_once:
                seen_once.remove(t)
                seen_atleast_twice.add(t)
            else:
                seen_once.add(t)
    logging.info("Rempved {} tokens".format(len(seen_once)))
    vocab.difference_update(seen_once)
    with open(vocab_filepath, "w+") as out_vocab:
        out_vocab.write("\n".join(vocab))


def remove_words_not_present(locations, vocab_filepath, config={}):
    text_files = [get_text_from_files(l) for l in locations]
    texts = []
    for files in text_files:
        texts.extend([strip_accents_ascii(x) for _, x in files])
    vocab_file = open(vocab_filepath, "r")
    vocab = set([line.strip('\n') for line in vocab_file.readlines()])
    orig_size = len(vocab)
    result_vocab = set()
    tok_re = re.compile(re_pattern_tok)
    for text in texts:
        toks = tok_re.findall(text)
        toks = [t.lower() for t in toks]
        for t in set(toks):
            if t in vocab:
                result_vocab.add(t)
                vocab.remove(t)
    logging.info("Removed {} toks".format(orig_size - len(result_vocab)))
    with open(vocab_filepath, "w+") as out_vocab:
        out_vocab.write("\n".join(result_vocab))


def set_up_vocab(locations, vocab_filepath, config={}):
    text_files = [get_text_from_files(l) for l in locations]
    texts = []
    for files in text_files:
        texts.extend([x for _, x in files])
    strip_accents = 'ascii' if "strip_accents" in config and config["strip_accents"] else None
    vectorizer = CountVectorizer(
        token_pattern=re_pattern_tok,
        strip_accents=strip_accents
    )
    vectorizer.fit(texts)
    words = list(vectorizer.vocabulary_.keys())
    order_1 = sorted(words, key=lambda x: word_frequency(x.split("_")[0], 'en'), reverse=True)
    alpha_regex = re.compile("[a-zA-Z_]+$")
    order_2 = sorted(order_1, key=lambda x: 0 if alpha_regex.match(x) else 1)
    with open(vocab_filepath, "w+") as out_vocab:
        out_vocab.write("\n".join(order_2))


def get_text_and_label(input_paths):
    texts = []
    ys = []
    for in_dir in input_paths:
        filename_text = sorted(get_text_from_files(in_dir), key=lambda x: int(x[0].split("_")[0]))
        texts.extend([x for _, x in filename_text])
        ys.extend([y.split(".")[0].split("_")[1] for y, _ in filename_text])
    return texts, ys


def to_svmlight_vectors(input_paths_train, input_paths_test, vocab_path, output_path_train, output_path_test,
                        config={}):
    train_texts, train_ys = get_text_and_label(input_paths_train)
    test_texts, test_ys = get_text_and_label(input_paths_test)

    vocab_file = open(vocab_path, "r")
    vocab = [line.strip('\n') for line in vocab_file.readlines()]

    x = 1
    vectorizer = CountVectorizer(
        # token_pattern='[a-zA-Z0-9{}]+_[a-zA-Z0-9{}]+'.format(string.punctuation, string.punctuation)
        token_pattern=r"(?u)\S+",
        vocabulary=vocab
    )

    train_vector = vectorizer.fit_transform(train_texts)
    test_vector = vectorizer.transform(test_texts)
    if "tfidf" in config:
        tf_vectorizer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tf_vectorizer.fit(train_vector)
        train_vector = tf_vectorizer.transform(train_vector)
        test_vector = tf_vectorizer.transform(test_vector)

    sklearn.datasets.dump_svmlight_file(train_vector, np.array([int(y) for y in train_ys]), output_path_train)
    sklearn.datasets.dump_svmlight_file(test_vector, np.array([int(y) for y in test_ys]), output_path_test)


if __name__ == "__main__":
    # variables to change
    vocab_path = "aclImdb 2/imdb_james_preprocess_no_pos.vocab"
    location_suffix = "_preprocess_no_pos"

    # consts
    train_locations = ["aclImdb 2/train/pos", "aclImdb 2/train/neg", ]
    test_locations = ["aclImdb 2/test/pos", "aclImdb 2/test/neg"]

    train_locations = [l + location_suffix for l in train_locations]
    test_locations = [l + location_suffix for l in test_locations]
    set_up_vocab(train_locations, vocab_path)

    to_svmlight_vectors(train_locations, vocab_path, "aclImdb 2/train/labeledBowNoPos.feat")

    to_svmlight_vectors(test_locations, vocab_path, "aclImdb 2/test/labeledBowNoPos.feat")
