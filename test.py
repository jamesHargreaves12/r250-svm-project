import os

TRAIN_DIR = "/Users/james_hargreaves/PycharmProjects/r250/aclImdb 2/train"
for valence in ["pos", "neg"]:
    dir_path = os.path.join("/Users/james_hargreaves/PycharmProjects/r250/aclImdb 2/train", valence)
    for file in os.listdir(dir_path):
        with open(os.path.join(dir_path, file), "r") as fp:
            print(fp.read())
        break
count = 0
with open("/Users/james_hargreaves/PycharmProjects/r250/aclImdb 2/train/labeledBow.feat", "r")as fp:
    for line in fp:
        count += 1
print(count)
print(line)

