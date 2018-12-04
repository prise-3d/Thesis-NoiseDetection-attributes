from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

import numpy as np
import pandas as pd

from ipfml import image_processing
from PIL import Image

import sys, os, getopt
import subprocess
import time

vector_size = 100
epochs = 100
batch_size = 24

input_shape = (vector_size, 1)
filename = "svd_model"

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def generate_model():

    model = Sequential()

    #model.add(Conv1D(128, (10), input_shape=input_shape))
    #model.add(Activation('relu'))

    #model.add(Conv1D(128, (10)))
    #model.add(Activation('relu'))

    #model.add(Conv1D(128, (10)))
    #model.add(Activation('relu'))
    #model.add(MaxPooling1D(pool_size=(2)))

    #model.add(Conv1D(64, (10)))
    #model.add(Activation('relu'))

    #model.add(Conv1D(64, (10)))
    #model.add(Activation('relu'))

    #model.add(Conv1D(64, (10)))
    #model.add(Activation('relu'))
    #model.add(MaxPooling1D(pool_size=(2)))

    #model.add(Conv1D(32, (10)))
    #model.add(Activation('relu'))

    #model.add(Conv1D(32, (10)))
    #model.add(Activation('relu'))

    #model.add(Conv1D(32, (10)))
    #model.add(Activation('relu'))
    #model.add(MaxPooling1D(pool_size=(2)))

    model.add(Flatten(input_shape=input_shape))

    #model.add(Dense(2048))
    #model.add(Activation('relu'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.3))

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', f1])

    return model

def main():

    if len(sys.argv) <= 1:
        print('Run with default parameters...')
        print('python save_model_result_in_md.py --data filename')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd", ["help=", "data="])
    except getopt.GetoptError:
        # print help information and exit:
        print('python save_model_result_in_md.py --data filename')
        sys.exit(2)
    for o, a in opts:
        if o == "-h":
            print('python save_model_result_in_md.py --data filename')
            sys.exit()
        elif o in ("-d", "--data"):
            p_datafile = a
        else:
            assert False, "unhandled option"

    ###########################
    # 1. Get and prepare data
    ###########################
    dataset_train = pd.read_csv(p_datafile + '.train', header=None, sep=";")
    dataset_test = pd.read_csv(p_datafile + '.test', header=None, sep=";")

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

    model = generate_model()
    model.summary()

    #######################
    # 3. Fit model : use of cross validation to fit model
    #######################

    # reshape input data
    x_dataset_train = np.array(x_dataset_train).reshape(len(x_dataset_train), vector_size, 1)
    x_dataset_test = np.array(x_dataset_test).reshape(len(x_dataset_test), vector_size, 1)

    model.fit(x_dataset_train, y_dataset_train, epochs=epochs, batch_size=batch_size, validation_split=0.20)

    score = model.evaluate(x_dataset_test, y_dataset_test, batch_size=batch_size)
    print(score)

if __name__== "__main__":
    main()
