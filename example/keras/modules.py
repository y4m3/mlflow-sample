import numpy as np
from keras.datasets import reuters
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical


def load_data(max_words, test_split_rate):
    (x_train, y_train), (x_test, y_test) = reuters.load_data(
        num_words=max_words, test_split=test_split_rate)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    return (x_train, y_train), (x_test, y_test)


def preprocess_features(x_train, x_test, max_words):
    print('Vectorizing sequence data...')
    tokenizer = Tokenizer(num_words=max_words)
    x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
    x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    return x_train, x_test


def preprocess_targets(y_train, y_test, num_classes):
    print('Convert class vector to binary class matrix '
          '(for use with categorical_crossentropy)')
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)
    return y_train, y_test


def model(max_words, num_classes):
    model = Sequential()
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model
