from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.utils import shuffle

import sklearn.svm as svm
from sklearn.externals import joblib

import numpy as np

import pandas as pd
from sklearn.metrics import accuracy_score

import sys, os, getopt

output_model_folder = './saved_models/'

def get_best_model(X_train, y_train):
    parameters = {'kernel':['rbf'], 'C': np.arange(1, 20)}
    svc = svm.SVC(gamma="scale")
    clf = GridSearchCV(svc, parameters, cv=5, scoring='accuracy', verbose=10)

    clf.fit(X_train, y_train)

    model = clf.best_estimator_

    return model


def main():

    if len(sys.argv) <= 1:
        print('Run with default parameters...')
        print('python svm_model_train.py --data xxxx --output xxxx')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:o", ["help=", "data=", "output="])
    except getopt.GetoptError:
        # print help information and exit:
        print('python svm_model_train.py --data xxxx --output xxxx')
        sys.exit(2)
    for o, a in opts:
        if o == "-h":
            print('python svm_model_train.py --data xxxx --output xxxx')
            sys.exit()
        elif o in ("-d", "--data"):
            p_data_file = a
        elif o in ("-o", "--output"):
            p_output = a
        else:
            assert False, "unhandled option"

    if not os.path.exists(output_model_folder):
        os.makedirs(output_model_folder)

    dataset = pd.read_csv(p_data_file, header=None, sep=";")

    # default first shuffle of data
    dataset = shuffle(dataset)
    
    # get dataset with equal number of classes occurences
    noisy_df = dataset[dataset.ix[:, 0] == 1]
    not_noisy_df = dataset[dataset.ix[:, 0] == 0]
    nb_not_noisy = len(not_noisy_df.index)

    final_df = pd.concat([not_noisy_df, noisy_df[0:nb_not_noisy]])
  
    # shuffle data another time
    final_df = shuffle(final_df)

    y_dataset = final_df.ix[:,0]
    x_dataset = final_df.ix[:,1:]

    X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.4, random_state=42)

    svm_model = get_best_model(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    print(str(accuracy_score(y_test, y_pred)) + '\n')

    joblib.dump(svm_model, output_model_folder + p_output + '.joblib') 

if __name__== "__main__":
    main()
