from sklearn.externals import joblib

import numpy as np

import pandas as pd
from sklearn.metrics import accuracy_score

import sys, os, getopt

output_model_folder = './saved_models/'

def main():

    if len(sys.argv) <= 1:
        print('Run with default parameters...')
        print('python smv_model_train.py --data xxxx.csv --model xxxx.joblib --output xxxx')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:o", ["help=", "data=", "model=", "output="])
    except getopt.GetoptError:
        # print help information and exit:
        print('python smv_model_train.py --data xxxx.csv --model xxxx.joblib --output xxxx')
        sys.exit(2)
    for o, a in opts:
        if o == "-h":
            print('python smv_model_train.py --data xxxx.csv --model xxxx.joblib --output xxxx')
            sys.exit()
        elif o in ("-d", "--data"):
            p_data_file = a
        elif o in ("-m", "--model"):
            p_model_file = a
        elif o in ("-o", "--output"):
            p_output = a
        else:
            assert False, "unhandled option"

    if not os.path.exists(output_model_folder):
        os.makedirs(output_model_folder)

    dataset = pd.read_csv(p_data_file, header=None, sep=";")

    y_dataset = dataset.ix[:,0]
    x_dataset = dataset.ix[:,1:]

    model = joblib.load(p_model_file) 

    y_pred = model.predict(x_dataset)

    print("Accuracy found %s " % str(accuracy_score(y_dataset, y_pred)))

if __name__== "__main__":
    main()
