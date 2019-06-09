# -*- coding: utf-8 -*-
import random
import os
import numpy as np
import pickle
import json

from datetime import datetime
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix

from logger import Logger
from config import ParamConfig
from utils import get_data_for_model


def split_train_test(X, Y):
    try:
        train_data, test_data, train_labels, test_labels = train_test_split(X, Y, test_size=0.33, random_state=42)
        Logger.log("Splitting of data set using train_test_split")
    except:
        train_size = int(len(X) * 0.8)
        Logger.log_error("The train_test_split failed, using manual splitting of data set")
        random.shuffle(X)
        train_data = X[:train_size]
        train_labels = Y[:train_size]

        test_data = X[train_size:]
        test_labels = Y[train_size:]

    Logger.log("Length of train_data is {}".format(len(train_data)))
    Logger.log("Length of test_data is  {}".format(len(test_data)))

    return train_data, test_data, train_labels, test_labels


def train_and_save_model(train_data, test_data, train_labels, test_labels):
    # define Tokenizer with Vocab Size
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data)

    x_train = tokenizer.texts_to_matrix(train_data, mode='tfidf')
    x_test = tokenizer.texts_to_matrix(test_data, mode='tfidf')

    encoder = LabelBinarizer()
    encoder.fit(train_labels)
    y_train = encoder.transform(train_labels)
    y_test = encoder.transform(test_labels)

    # y_train = train_labels
    # y_test = test_labels

    print(y_train[0])
    print(x_train.shape)
    print(y_train.shape)
    vocab_size = x_train.shape[1]

    model = Sequential()
    model.add(Dense(512, input_shape=(vocab_size,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(config.num_labels))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=config.batch_size,
                        epochs=config.num_epochs,
                        verbose=1,
                        validation_split=0.1)

    # creates a HDF5 file
    model.model.save(config.model_name)

    # Save Tokenizer i.e. Vocabulary
    with open(config.tokenizer_name, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    score = model.evaluate(x_test, y_test,
                           batch_size=config.batch_size, verbose=1)

    print('Test loss: {}'.format(score[0]))
    print('Test accuracy:{}'.format(score[1]))

    labels_ = encoder.classes_

    encoded_label = dict()
    for i in range(len(labels_)):
        encoded_label[i] = str(labels_[i])
    with open(config.encoder_name, 'w') as fp:
        json.dump(encoded_label, fp)

    y_pred = model.predict(x_test)
    Logger.log(encoded_label)
    Logger.log("Confusion Matrix")
    Logger.log(labels_)
    cnf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    for i in range(len(labels_)):
        Logger.log(cnf_matrix[i])


if __name__ == "__main__":
    start = datetime.now()
    config = ParamConfig()
    data, label = get_data_for_model(config.train_preprocessed_path)
    train_data, test_data, train_labels, test_labels = split_train_test(data, label)
    train_and_save_model(train_data, test_data, train_labels, test_labels)
    Logger.log("Training finished in : {}".format(datetime.now()-start))
