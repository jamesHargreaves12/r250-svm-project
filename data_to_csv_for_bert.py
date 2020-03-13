import pandas as pd

from doc2vec_test import get_docs
from preprocess_text_files import preprocess

train_docs = get_docs('aclImdb 2/train/neg', include_lab=True) + get_docs('aclImdb 2/train/pos', include_lab=True)
train = []
for doc,lab in train_docs:
    train.append((preprocess(doc, config={"remove_punc":True}),lab))
df = pd.DataFrame(train)
df.to_csv('csv_files/train_2.csv', index=False, header=None)

test_docs = get_docs('aclImdb 2/test/neg', include_lab=True) + get_docs('aclImdb 2/test/pos', include_lab=True)
test = []
for doc,lab in train_docs:
    test.append((preprocess(doc, config={"remove_punc":True}),lab))

df = pd.DataFrame(test)
df.to_csv('csv_files/test_2.csv', index=False, header=None)
