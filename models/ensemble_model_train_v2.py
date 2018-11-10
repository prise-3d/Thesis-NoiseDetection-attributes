from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

import sklearn.svm as svm
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import cross_val_score

import numpy as np
import pandas as pd
import sys, os, getopt

saved_models_folder = 'saved_models'
current_dirpath = os.getcwd()
output_model_folder = os.path.join(current_dirpath, saved_models_folder)

def get_best_model(X_train, y_train):
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gammas = [0.001, 0.01, 0.1, 1, 5, 10, 100]
    param_grid = {'kernel':['rbf'], 'C': Cs, 'gamma' : gammas}

    svc = svm.SVC(probability=True)
    clf = GridSearchCV(svc, param_grid, cv=10, scoring='accuracy', verbose=10)

    clf.fit(X_train, y_train)

    model = clf.best_estimator_

    return model


def main():

    if len(sys.argv) <= 1:
        print('Run with default parameters...')
        print('python ensemble_model_train_v2.py --data xxxx --output xxxx')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:o", ["help=", "data=", "output="])
    except getopt.GetoptError:
        # print help information and exit:
        print('python ensemble_model_train_v2.py --data xxxx --output xxxx')
        sys.exit(2)
    for o, a in opts:
        if o == "-h":
            print('python ensemble_model_train_v2.py --data xxxx --output xxxx')
            sys.exit()
        elif o in ("-d", "--data"):
            p_data_file = a
        elif o in ("-o", "--output"):
            p_output = a
        else:
            assert False, "unhandled option"

    if not os.path.exists(output_model_folder):
        os.makedirs(output_model_folder)

    # 1. Get and prepare data
    dataset_train = pd.read_csv(p_data_file + '.train', header=None, sep=";")
    dataset_test = pd.read_csv(p_data_file + '.test', header=None, sep=";")

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
    # 2. Construction of the model : Ensemble model structure
    #######################

    svm_model = get_best_model(y_dataset_train, y_dataset_train)
    knc_model = KNeighborsClassifier(n_neighbors=2)
    gbc_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    lr_model = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=1)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=1)

    ensemble_model = VotingClassifier(estimators=[
       ('lr', lr_model),
       ('knc', knc_model),
       ('gbc', gbc_model),
       ('svm', svm_model),
       ('rf', rf_model)],
       voting='soft', weights=[1, 1, 1, 1, 1])


    #######################
    # 3. Fit model : use of cross validation to fit model
    #######################
    print("-------------------------------------------")
    print("Train dataset size: ", final_df_train_size)
    ensemble_model.fit(x_dataset_train, y_dataset_train)
    val_scores = cross_val_score(ensemble_model, x_dataset_train, y_dataset_train, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (val_scores.mean(), val_scores.std() * 2))

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

    y_test_model = ensemble_model.predict(X_test)
    y_val_model = ensemble_model.predict(X_val)

    val_accuracy = accuracy_score(y_val, y_val_model)
    test_accuracy = accuracy_score(y_test, y_test_model)

    val_f1 = f1_score(y_val, y_val_model)
    test_f1 = f1_score(y_test, y_test_model)

    ###################
    # 5. Output : Print and write all information in csv
    ###################

    print("Validation dataset size ", val_set_size)
    print("Validation: ", val_accuracy)
    print("Validation F1: ", val_f1)
    print("Test dataset size ", test_set_size)
    print("Test: ", val_accuracy)
    print("Test F1: ", test_f1)

    ##################
    # 6. Save model : create path if not exists
    ##################

    # create path if not exists
    if not os.path.exists(saved_models_folder):
        os.makedirs(saved_models_folder)

    joblib.dump(ensemble_model, output_model_folder + '/' +  p_output + '.joblib')

if __name__== "__main__":
    main()
