import os

import sklearn
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import word_tokenize
import numpy as np
from tqdm import tqdm


def get_docs(in_path, include_lab=False):
    in_list = os.listdir(in_path)
    results = []
    for file_path in in_list:
        with open(os.path.join(in_path, file_path), "r") as in_file:
            text = in_file.read()
            if include_lab:
                results.append((text, int(file_path.split("_")[1].split(".")[0])))
            else:
                results.append(text)
    return results


def train_embeddings(model_name, vector_size=100, window=5, tag_sent=False):
    if tag_sent:
        docs = get_docs("aclImdb 2/train/pos") + get_docs("aclImdb 2/train/neg")
        tokens = [word_tokenize(x) for x in docs]
        documents = [TaggedDocument(doc, [1 if i < 12500 else 0]) for i, doc in enumerate(tokens)]
    else:
        docs = get_docs("aclImdb 2/train/pos") + get_docs("aclImdb 2/train/neg") + get_docs(
            "aclImdb 2/test/pos") + get_docs("aclImdb 2/test/neg")
        tokens = [word_tokenize(x) for x in docs]
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokens)]
    model = Doc2Vec(documents, vector_size=vector_size, window=window, min_count=1, workers=4)
    model.save(os.path.join("models", model_name))


def build_embeddings(model_name, output_dir):
    model = Doc2Vec.load(os.path.join("models", model_name))
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    for doc_path in ["train", "test"]:
        emeddings = []
        labels = []
        for valence in ["pos", "neg"]:
            print(doc_path)
            in_path = os.path.join("aclImdb 2", doc_path, valence)
            in_list = os.listdir(os.path.join(in_path))
            for file_path in tqdm(in_list):
                with open(os.path.join(in_path, file_path), "r") as in_file:
                    text = in_file.read()
                    tokens = word_tokenize(text)
                    vector = model.infer_vector(tokens)
                    emeddings.append(vector)
                    labels.append(int(file_path.split("_")[1].split(".")[0]))
        out_path = os.path.join("doc2vec-embeddings/{}".format(output_dir), doc_path + ".feat")
        parent_loc = os.path.abspath(os.path.join(out_path, os.pardir))
        if not os.path.exists(parent_loc):
            os.makedirs(parent_loc)
        sklearn.datasets.dump_svmlight_file(np.array(emeddings), np.array(labels), out_path)


if __name__ == "__main__":
    fname = "doc2vec_medium_tagged"
    train_embeddings(fname, vector_size=1000, tag_sent=True)
    build_embeddings(fname, "medium_vector_tag")
