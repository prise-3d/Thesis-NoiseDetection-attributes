from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

from ipfml import processing
from PIL import Image

import sys, os, getopt
import subprocess
import time

from modules.utils import config as cfg

threshold_map_folder        = cfg.threshold_map_folder
threshold_map_file_prefix   = cfg.threshold_map_folder + "_"

markdowns_folder            = cfg.models_information_folder
final_csv_model_comparisons = cfg.csv_model_comparisons_filename
models_name                 = cfg.models_names_list

zones                       = cfg.zones_indices

current_dirpath = os.getcwd()


def main():

    if len(sys.argv) <= 1:
        print('Run with default parameters...')
        print('python save_model_result_in_md.py --interval "0,20" --model path/to/xxxx.joblib --mode ["svd", "svdn", "svdne"] --metric ["lab", "mscn"]')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ht:m:o:l", ["help=", "interval=", "model=", "mode=", "metric="])
    except getopt.GetoptError:
        # print help information and exit:
        print('python save_model_result_in_md.py --interval "xx,xx" --model path/to/xxxx.joblib --mode ["svd", "svdn", "svdne"] --metric ["lab", "mscn"]')
        sys.exit(2)
    for o, a in opts:
        if o == "-h":
            print('python save_model_result_in_md.py --interval "xx,xx" --model path/to/xxxx.joblib --mode ["svd", "svdn", "svdne"] --metric ["lab", "mscn"]')
            sys.exit()
        elif o in ("-t", "--interval"):
            p_interval = list(map(int, a.split(',')))
        elif o in ("-m", "--model"):
            p_model_file = a
        elif o in ("-o", "--mode"):
            p_mode = a

            if p_mode != 'svdn' and p_mode != 'svdne' and p_mode != 'svd':
                assert False, "Mode not recognized"
        elif o in ("-m", "--metric"):
            p_metric = a
        else:
            assert False, "unhandled option"


    # call model and get global result in scenes

    begin, end = p_interval

    bash_cmd = "bash testModelByScene_maxwell.sh '" + str(begin) + "' '" + str(end) + "' '" + p_model_file + "' '" + p_mode + "' '" + p_metric + "'"

    print(bash_cmd)

    ## call command ##
    p = subprocess.Popen(bash_cmd, stdout=subprocess.PIPE, shell=True)

    (output, err) = p.communicate()

    ## Wait for result ##
    p_status = p.wait()

    if not os.path.exists(markdowns_folder):
        os.makedirs(markdowns_folder)

    # get model name to construct model
    md_model_path = os.path.join(markdowns_folder, p_model_file.split('/')[-1].replace('.joblib', '.md'))

    with open(md_model_path, 'w') as f:
        f.write(output.decode("utf-8"))

        # read each threshold_map information if exists
        model_map_info_path = os.path.join(threshold_map_folder, p_model_file.replace('saved_models/', ''))

        if not os.path.exists(model_map_info_path):
            f.write('\n\n No threshold map information')
        else:
            maps_files = os.listdir(model_map_info_path)

            # get all map information
            for t_map_file in maps_files:

                file_path = os.path.join(model_map_info_path, t_map_file)
                with open(file_path, 'r') as map_file:

                    title_scene =  t_map_file.replace(threshold_map_file_prefix, '')
                    f.write('\n\n## ' + title_scene + '\n')
                    content = map_file.readlines()

                    # getting each map line information
                    for line in content:
                        f.write(line)

        f.close()

    # Keep model information to compare
    current_model_name = p_model_file.split('/')[-1].replace('.joblib', '')

    # Prepare writing in .csv file
    output_final_file_path = os.path.join(markdowns_folder, final_csv_model_comparisons)
    output_final_file = open(output_final_file_path, "a")

    print(current_model_name)
    # reconstruct data filename
    for name in models_name:
        if name in current_model_name:
            current_data_file_path = os.path.join('data', current_model_name.replace(name, 'data_maxwell'))

    model_scores = []

    ########################
    # 1. Get and prepare data
    ########################
    dataset_train = pd.read_csv(current_data_file_path + '.train', header=None, sep=";")
    dataset_test = pd.read_csv(current_data_file_path + '.test', header=None, sep=";")

    # default first shuffle of data
    dataset_train = shuffle(dataset_train)
    dataset_test = shuffle(dataset_test)

    # get dataset with equal number of classes occurences
    noisy_df_train = dataset_train[dataset_train.ix[:, 0] == 1]
    not_noisy_df_train = dataset_train[dataset_train.ix[:, 0] == 0]
    nb_noisy_train = len(noisy_df_train.index)

    noisy_df_test = dataset_test[dataset_test.ix[:, 0] == 1]
    not_noisy_df_test = dataset_test[dataset_test.ix[:, 0] == 0]
    nb_noisy_test = len(noisy_df_test.index)

    final_df_train = pd.concat([not_noisy_df_train[0:nb_noisy_train], noisy_df_train])
    final_df_test = pd.concat([not_noisy_df_test[0:nb_noisy_test], noisy_df_test])

    # shuffle data another time
    final_df_train = shuffle(final_df_train)
    final_df_test = shuffle(final_df_test)

    final_df_train_size = len(final_df_train.index)
    final_df_test_size = len(final_df_test.index)

    # use of the whole data set for training
    x_dataset_train = final_df_train.ix[:,1:]
    x_dataset_test = final_df_test.ix[:,1:]

    y_dataset_train = final_df_train.ix[:,0]
    y_dataset_test = final_df_test.ix[:,0]

    #######################
    # 2. Getting model
    #######################

    model = joblib.load(p_model_file)

    #######################
    # 3. Fit model : use of cross validation to fit model
    #######################
    model.fit(x_dataset_train, y_dataset_train)
    val_scores = cross_val_score(model, x_dataset_train, y_dataset_train, cv=5)

    ######################
    # 4. Test : Validation and test dataset from .test dataset
    ######################

    # we need to specify validation size to 20% of whole dataset
    val_set_size = int(final_df_train_size/3)
    test_set_size = val_set_size

    total_validation_size = val_set_size + test_set_size

    if final_df_test_size > total_validation_size:
        x_dataset_test = x_dataset_test[0:total_validation_size]
        y_dataset_test = y_dataset_test[0:total_validation_size]

    X_test, X_val, y_test, y_val = train_test_split(x_dataset_test, y_dataset_test, test_size=0.5, random_state=1)

    y_test_model = model.predict(X_test)
    y_val_model = model.predict(X_val)

    val_accuracy = accuracy_score(y_val, y_val_model)
    test_accuracy = accuracy_score(y_test, y_test_model)

    y_train_model = model.predict(x_dataset_train)

    train_f1 = f1_score(y_dataset_train, y_train_model)
    train_recall = recall_score(y_dataset_train, y_train_model)
    train_roc_auc = roc_auc_score(y_dataset_train, y_train_model)

    val_f1 = f1_score(y_val, y_val_model)
    val_recall = recall_score(y_val, y_val_model)
    val_roc_auc = roc_auc_score(y_val, y_val_model)

    test_f1 = f1_score(y_test, y_test_model)
    test_recall = recall_score(y_test, y_test_model)
    test_roc_auc = roc_auc_score(y_test, y_test_model)

    # stats of all dataset
    all_x_data = pd.concat([x_dataset_train, X_test, X_val])
    all_y_data = pd.concat([y_dataset_train, y_test, y_val])

    all_y_model = model.predict(all_x_data)
    all_accuracy = accuracy_score(all_y_data, all_y_model)
    all_f1_score = f1_score(all_y_data, all_y_model)
    all_recall_score = recall_score(all_y_data, all_y_model)
    all_roc_auc_score = roc_auc_score(all_y_data, all_y_model)

    # stats of dataset sizes
    total_samples = final_df_train_size + val_set_size + test_set_size

    model_scores.append(final_df_train_size)
    model_scores.append(val_set_size)
    model_scores.append(test_set_size)

    model_scores.append(final_df_train_size / total_samples)
    model_scores.append(val_set_size / total_samples)
    model_scores.append(test_set_size / total_samples)

    # add of scores
    model_scores.append(val_scores.mean())
    model_scores.append(val_accuracy)
    model_scores.append(test_accuracy)
    model_scores.append(all_accuracy)

    model_scores.append(train_f1)
    model_scores.append(train_recall)
    model_scores.append(train_roc_auc)

    model_scores.append(val_f1)
    model_scores.append(val_recall)
    model_scores.append(val_roc_auc)

    model_scores.append(test_f1)
    model_scores.append(test_recall)
    model_scores.append(test_roc_auc)

    model_scores.append(all_f1_score)
    model_scores.append(all_recall_score)
    model_scores.append(all_roc_auc_score)

    # TODO : improve...
    # check if it's always the case...
    nb_zones = current_data_file_path.split('_')[7]

    final_file_line = current_model_name + '; ' + str(end - begin) + '; ' + str(begin) + '; ' + str(end) + '; ' + str(nb_zones) + '; ' + p_metric + '; ' + p_mode

    for s in model_scores:
        final_file_line += '; ' + str(s)

    output_final_file.write(final_file_line + '\n')


if __name__== "__main__":
    main()
