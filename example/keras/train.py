'''Keras example
Base example:
    https://github.com/mlflow/mlflow/blob/master/examples/keras/train.py
'''
import tempfile

import mlflow
import mlflow.keras
import numpy as np
from keras.datasets import reuters
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical


# functions
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


if __name__ == '__main__':
    # start mlflow session
    mlflow.start_run(run_name='keras_reuters')

    # set mlflow keras autolog
    '''mlflow.keras.autolog func logs following contents (mlflow==1.2.0).
    - Parameteres
    - epsilon
    - learning_rate
    - num_layers
    - optimizer_name
    - Metrics
    - acc
    - loss
    - val_acc
    - val_loss
    - Tags
    - model.summary()
    - Artifacts
    - envrionments info
    - model
    '''
    mlflow.keras.autolog()

    # load data hyper-prams
    max_words = 1000
    split_rate = 0.2

    # load data
    (x_train, y_train), (x_test, y_test) = load_data(max_words, split_rate)

    # preprocess
    num_classes = np.max(y_train) + 1
    x_train, x_test = preprocess_features(x_train, x_test, max_words)
    y_train, y_test = preprocess_targets(y_train, y_test, num_classes)

    # building model
    model = model(max_words, num_classes)

    # model hyper-params
    batch_size = 32
    epochs = 10
    loss_func = 'categorical_crossentropy'
    optimizer = 'adam'
    metrics = 'accuracy'

    # model compile
    model.compile(loss=loss_func, optimizer=optimizer, metrics=[metrics])

    # learning
    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1)

    # evaluate
    score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

    # logging params (following params does not log mlflow.keras.autolog())
    mlflow.log_param('batch_size', batch_size)
    mlflow.log_param('loss', loss_func)
    mlflow.log_param('max_epochs', epochs)

    # export artifact (data shape tsv)
    output_dir = tempfile.mkdtemp()
    with open(f'{output_dir}/data.shape.tsv', 'w') as f:
        f.write('name\tshape\n'
                f'x_train\t{x_train.shape}\n'
                f'y_train\t{y_train.shape}\n'
                f'x_test\t{x_test.shape}\n'
                f'y_test\t{y_test.shape}')
    mlflow.log_artifact(f'{output_dir}/data.shape.tsv')

    # end mlflow session
    mlflow.end_run()
