import tempfile

import mlflow
import mlflow.keras
import numpy as np

import modules

# set mlflow keras autolog
'''
mlflow.keras.autolog func logs following contents (mlflow==1.2.0).
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
(x_train, y_train), (x_test, y_test) = modules.load_data(max_words, split_rate)

# preprocess
num_classes = np.max(y_train) + 1
x_train, x_test = modules.preprocess_features(x_train, x_test, max_words)
y_train, y_test = modules.preprocess_targets(y_train, y_test, num_classes)

# building model
model = modules.model(max_words, num_classes)

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
