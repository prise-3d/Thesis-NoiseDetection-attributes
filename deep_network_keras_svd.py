from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K

from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd

from ipfml import processing
import modules.utils.config as cfg

from PIL import Image

import sys, os
import argparse
import json

import subprocess
import time

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

def generate_model(input_shape):

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

    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', f1])

    return model

def main():

    parser = argparse.ArgumentParser(description="Process deep_network_keras_svd.py parameters")

    parser.add_argument('--data', type=str, help='Data filename prefix to access train and test dataset')
    parser.add_argument('--output', type=str, help='Name of filename to save model into')
    parser.add_argument('--size', type=int, help='Size of input data vector')

    args = parser.parse_args()

    p_datafile = args.data
    p_output_filename = args.output
    p_vector_size = args.size

    epochs = 10
    batch_size = cfg.keras_batch

    input_shape = (p_vector_size, 1)

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

    model = generate_model(input_shape)
    model.summary()
    #model = KerasClassifier(build_fn=model, epochs=cfg.keras_epochs, batch_size=cfg.keras_batch, verbose=0)

    #######################
    # 3. Fit model : use of cross validation to fit model
    #######################

    # reshape input data
    x_dataset_train = np.array(x_dataset_train).reshape(len(x_dataset_train), p_vector_size, 1)
    x_dataset_test = np.array(x_dataset_test).reshape(len(x_dataset_test), p_vector_size, 1)

    model.fit(x_dataset_train, y_dataset_train, validation_split=0.20, epochs=cfg.keras_epochs, batch_size=cfg.keras_batch)

    score = model.evaluate(x_dataset_test, y_dataset_test, batch_size=batch_size)

    if not os.path.exists(cfg.saved_models_folder):
        os.makedirs(cfg.saved_models_folder)

    # save the model into HDF5 file
    model_output_path = os.path.join(cfg.saved_models_folder, p_output_filename + '.json')
    json_model_content = model.to_json()

    with open(model_output_path, 'w') as f:
        print("Model saved into ", model_output_path)
        json.dump(json_model_content, f, indent=4)

    model.save_weights(model_output_path.replace('.json', '.h5'))

    # Save results obtained from model
    y_test_prediction = model.predict(x_dataset_test)
    print("Metrics : ", model.metrics_names)
    print("Prediction : ", score)
    print("ROC AUC : ", roc_auc_score(y_dataset_test, y_test_prediction))


if __name__== "__main__":
    main()
