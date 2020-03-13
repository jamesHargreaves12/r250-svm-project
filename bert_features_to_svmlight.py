import sklearn.datasets
import pandas as pd
import numpy as np
file_path_train = 'csv_files/train_2.csv'
file_path_test = 'csv_files/test_2.csv'

df_embeddings = pd.read_csv("bert_output/bert_output_3.txt", delimiter=' ', header=None)
df_train = pd.read_csv(file_path_train, header=None)
df_test = pd.read_csv(file_path_test, header=None)

print(df_embeddings.shape)
print(df_train.shape)
print(df_test.shape)
sklearn.datasets.dump_svmlight_file(np.array(df_embeddings[:25000]), np.array(df_train[1]), 'bert_embedings/v3/train.feat')
sklearn.datasets.dump_svmlight_file(np.array(df_embeddings[25000:]), np.array(df_test[1]), 'bert_embedings/v3/test.feat')
