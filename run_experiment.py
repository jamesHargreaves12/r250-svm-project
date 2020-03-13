import os
import pickle
import sys
from itertools import product

import numpy as np
from sklearn.datasets import load_svmlight_file
import yaml
import shutil
import logging

# Constants
import bayes_opt_attempt
from bayes_opt_attempt import get_optimum_hyperparams
from preprocess_text_files import run_preprocess, run_preprocess_v2
from run_svm_class_analysis import get_trained_model, get_class_accuracies
from sparse_bow_vector import set_up_vocab, to_svmlight_vectors, remove_words_present_in_one_doc, \
    remove_words_not_present

base_config_path = "config"
base_experiment_path = "experiments_2"


def osjoin(path1, path2):
    return os.path.join(path1, path2)


def copy_dir_to_dir(original, end):
    for file_name in os.listdir(original):
        shutil.copyfile(osjoin(original, file_name), osjoin(end, file_name))


def save_config(config, experiment_folder):
    yaml.dump(config, stream=open(osjoin(experiment_folder, "config.yaml"), "w+"))


def run_experiment(config_file):
    config_path = osjoin(base_config_path, config_file)
    config = yaml.load(open(config_path, "r"))

    experiment_folder = osjoin(base_experiment_path, config["experiment_name"])
    # This should be reenabled after testing
    # assert (not (os.path.exists(experiment_folder)))
    if os.path.exists(experiment_folder):
        config = yaml.load(open(osjoin(experiment_folder, "config.yaml"), "r"))
    else:
        os.mkdir(experiment_folder)
        save_config(config, experiment_folder)

    logging.basicConfig(filename=osjoin(experiment_folder, "log.txt"), level=logging.INFO,
                        format='%(asctime)s %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(config)

    # preprocess files
    if "preprocess_complete" not in config:
        logging.info("Start Preprocessing")
        config["preprocess_outpath_format"] = osjoin(experiment_folder, "{}/{}.txt")
        for x, y in product(["train", "test"], ["pos", "neg"]):
            if "preprocess_file_location" not in config:
                run_preprocess_v2(osjoin('aclImdb 2', x + "/" + y), config["preprocess_outpath_format"].format(x, y),
                                  config)
            else:
                logging.info("Loading preprocessed files from {}".format(config['preprocess_file_location']))
                shutil.copyfile(osjoin(config['preprocess_file_location'], x + "/" + y + ".txt"),
                                config["preprocess_outpath_format"].format(x, y))
        config["preprocess_complete"] = True
        logging.info("End Preprocessing")

        save_config(config, experiment_folder)

    # Get BOW
    if "bow_test_feat" not in config:
        config["bow_test_feat"] = osjoin(experiment_folder, "test/labeledBow.feat")
    elif not config["bow_test_feat"].startswith(experiment_folder):
        config["bow_test_feat"] = osjoin(experiment_folder, config["bow_test_feat"])

    if "bow_train_feat" not in config:
        config["bow_train_feat"] = osjoin(experiment_folder, "train/labeledBow.feat")
    elif not config["bow_train_feat"].startswith(experiment_folder):
        config["bow_train_feat"] = osjoin(experiment_folder, config["bow_train_feat"])

    if "premade_embedding_loc" in config and config["premade_embedding_loc"]:
        for doc_path in ["train", "test"]:
            inpath = os.path.join(config["premade_embedding_loc"], doc_path + ".feat")
            out_path = config["bow_{}_feat".format(doc_path)]
            parent_loc = os.path.abspath(os.path.join(out_path, os.pardir))
            if not os.path.exists(parent_loc):
                os.makedirs(parent_loc)

            shutil.copyfile(inpath, out_path)
    elif "bow_complete" not in config:
        logging.info("Start BOW")
        config["bow_vocab_path"] = osjoin(experiment_folder, "imdb.vocab")
        config["bow_train_locations"] = [
            osjoin(experiment_folder, "train/pos.txt"),
            osjoin(experiment_folder, "train/neg.txt")
        ]
        config["bow_test_locations"] = [
            osjoin(experiment_folder, "test/pos.txt"),
            osjoin(experiment_folder, "test/neg.txt")
        ]

        logging.info("Create Vocab")
        if "premade_vocab_path" not in config:
            set_up_vocab(config["bow_train_locations"], config["bow_vocab_path"], config)
        else:
            logging.info("Loading config from {}".format(config["premade_vocab_path"]))
            shutil.copyfile(config["premade_vocab_path"], config["bow_vocab_path"])

        if "remove_infreq_words" in config:
            logging.info("Removing words only present once")
            remove_words_present_in_one_doc(config["bow_train_locations"], config["bow_vocab_path"], config)

        if "remove_words_not_in_test" in config:
            logging.info("Removing words not present in test")
            remove_words_not_present(config["bow_test_locations"], config["bow_vocab_path"], config)

        logging.info("Create Feats")
        to_svmlight_vectors(
            config["bow_train_locations"],
            config["bow_test_locations"],
            config["bow_vocab_path"],
            config["bow_train_feat"],
            config["bow_test_feat"],
            config
        )

        config["bow_complete"] = True
        logging.info("End BOW")

        save_config(config, experiment_folder)

    # Bayes Optimisation
    if "optimisation_complete" not in config:
        logging.info("Start Parameter Optimisation")
        config["opt_kernel"] = 'linear'
        max_point = None
        max_kernel = None
        max_val = -1
        for kernel in ['linear', 'rbf', 'sigmoid', 'poly']:
            logging.info("Testing {} kernel".format(kernel))
            max_c = int(config["max_c"]) if "max_c" in config else 7
            if kernel == 'linear':
                config["opt_limits"] = {"c": [-5, max_c]}
            else:
                config["opt_limits"] = {"c": [-5, max_c], "gamma": [-5, 7]}
            logging.info("Setting up Training Params")
            bayes_opt_attempt.setup(config["bow_train_feat"], config["opt_kernel"])
            logging.info("Optimising")
            point, val = get_optimum_hyperparams(config["opt_kernel"], config["opt_limits"], False, iterations=20)
            logging.info("{} at {} => {}".format(kernel, point, val))
            if val > max_val:
                max_point = point
                max_kernel = kernel
                max_val = val
        logging.info("Best kernel = {}".format(max_kernel))
        config["opt_max_point"], config["opt_max_val"] = [float(max_point[0])], float(max_val)
        if max_point.shape[0] > 1:
            config["opt_max_point"] = [float(max_point[0]), float(max_point[1])]

        logging.info("Maximum Performance at {} => {}".format(config["opt_max_point"], config["opt_max_val"]))
        logging.info("End Parameter Optimisation")
        config["optimisation_complete"] = True
        save_config(config, experiment_folder)

    logging.info("Loading Dataset")
    X_train, y_train = load_svmlight_file(config["bow_train_feat"])
    X_test, y_test = load_svmlight_file(config["bow_test_feat"], n_features=X_train.shape[1])
    y_train = [1 if a > 5 else -1 for a in y_train]
    logging.info("Dataset Loaded")

    # Training models
    if "training_model_complete" in config:
        model_path = osjoin(experiment_folder, "model.pickle")
        file = open(model_path, "rb")
        model = pickle.load(file)

    else:
        model_path = osjoin(experiment_folder, "model.pickle")
        logging.info("Start Training Model")
        model = get_trained_model(config["opt_max_point"], config["opt_kernel"], X_train, y_train)
        logging.info("End Training Model")
        file = open(model_path, "wb")
        pickle.dump(model, file)
        config["training_model_complete"] = True
        save_config(config, experiment_folder)

    # Testing models
    logging.info("Testing")
    pred_y = model.predict(X_test)
    overall_acc, class_accs = get_class_accuracies(pred_y, y_test)
    logging.info("Overall = {}".format(overall_acc))
    for key in sorted(class_accs.keys()):
        logging.info("Class: {}, Acc: {}".format(key, class_accs[key]))

    logging.info("Predicting even number of each class")
    pred_y = model.decision_function(X_test)
    logging.info("Model Made prediction")
    neg_vals = pred_y.argsort()[:X_test.shape[0] // 2]
    u_labs = np.ones(X_test.shape[0])
    u_labs[neg_vals] = -1
    overall_acc, class_accs = get_class_accuracies(u_labs, y_test)
    logging.info("Overall = {}".format(overall_acc))
    for key in sorted(class_accs.keys()):
        logging.info("Class: {}, Acc: {}".format(key, class_accs[key]))

    logging.info("End Testing Model")


if __name__ == "__main__":
    # for yaml_file in os.listdir(base_config_path):
    # if ".yaml" in yaml_file:
    #     config_path = osjoin(base_config_path, yaml_file)
    #     config = yaml.load(open(config_path, "r"))
    #     dir_name = config['experiment_name']
    #     if dir_name not in os.listdir(base_experiment_path):
    #         print("Running Experiment by the name:", dir_name)
    #         run_experiment(yaml_file)

    # config_file = "doc2vec_embeddings_3000_tag.yaml"
    config_file = "vanilla.yaml"
    run_experiment(config_file)
