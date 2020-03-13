import numpy as np
import pandas as pd
import torch
import transformers as ppb
from tqdm import tqdm

file_path_train = 'csv_files/train_2.csv'
file_path_test = 'csv_files/test_2.csv'
df_train = pd.read_csv(file_path_train, header=None)
df_test = pd.read_csv(file_path_test, header=None)
df = pd.concat((df_train, df_test))
df[1] = (df[1] - 1) // 5

# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (
    ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)


def populate_out_file(start, filename):
    batch = df[start:start + batch_size]
    tokenized = batch[0].apply((lambda x: tokenizer.encode(x, max_length=512, pad_to_max_length=True)))
    padded = np.array([i + [0] * (512 - len(i)) for i in tokenized.values])
    attention_mask = np.where(padded != 0, 1, 0)
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:, 0, :].numpy()

    with open(filename, "a+") as fp:
        for f in features:
            fp.write(' '.join([str(x) for x in list(f)]) + '\n')


batch_size = 100
out_file = "bert_output/bert_output_3.txt"
already_done = len(open(out_file, "r+").readlines())
for start in tqdm(range(already_done, 50000, batch_size)):
    populate_out_file(start, out_file)
    start += batch_size
