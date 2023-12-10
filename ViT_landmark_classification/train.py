from datetime import date
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD

from data import Data
from models import RESNET50, ViT, boosted_ensemble_model
from utils import Config, Parameters
import pickle


def define_paths(model_name: str,
                 heading: int = None) -> tuple:
    weights_dir = Path(f'{Config.WEIGHTS_PATH}{model_name}_{date.today()}')
    weights_dir.mkdir(exist_ok=True, parents=True)

    if heading is not None:
        filepath = f'{Config.WEIGHTS_PATH}{model_name}_{date.today()}/{heading}' + '.{epoch:02d}-{loss:.2f}.hdf5'
        logdir = f'{Config.PATH}logs/{model_name}_{heading}_{date.today()}'
    else:
        filepath = f'{Config.WEIGHTS_PATH}{model_name}_{date.today()}/' + '.{epoch:02d}-{loss:.2f}.hdf5'
        logdir = f'{Config.PATH}logs/{model_name}_{date.today()}'

    return filepath, logdir


def callbacks(model_name: str,
              heading: int = None) -> list:
    filepath, logdir = define_paths(model_name, heading)
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=Parameters.verbose, save_weights_only=True,
                                 save_best_only=True, mode='auto')
    tensor_board = TensorBoard(log_dir=logdir)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=Parameters.early_stop_patience,
                               verbose=Parameters.verbose)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=Parameters.reduce_lr_factor,
                                  patience=Parameters.reduce_lr_patience, min_lr=Parameters.min_lr)
    return [checkpoint, tensor_board, early_stop, reduce_lr]


def compile_SGD(model,
                lr: float = Parameters.learning_rate,
                momentum: float = Parameters.momentum):
    return model.compile(optimizer=SGD(learning_rate=lr, momentum=momentum), loss='categorical_crossentropy', metrics=['accuracy'])


def train(model,
          train_data: tf.data.Dataset,
          val_data: tf.data.Dataset,
          num_epochs: int,
          model_name: str,
          heading=None):
    return model.fit(train_data, validation_data=val_data, epochs=num_epochs, callbacks=callbacks(model_name, heading))


def train_ViT(model_name: str,
              num_epochs: int = Parameters.num_epochs):
    data = Data(validation_split=0.15, image_size=(Parameters.image_size, Parameters.image_size), batch_size=8, seed=123, label_mode='categorical')
    input_shape = data.image_size + (3,)
    ds = data.load_train('data')
    x_train, y_train = [], []

    for images, labels in ds.unbatch():
        x_train.append(images.numpy())
        y_train.append(labels.numpy())

    x_train, y_train = np.array(x_train), np.array(y_train)
    model = ViT(input_shape, x_train)
    compile_SGD(model)

    return model.fit(x=x_train, y=y_train, epochs=num_epochs, validation_split=0.15, callbacks=callbacks(model_name))


if __name__ == '__main__':
    # Creating and training the model
    history = train_ViT('ViT')
    # Save the history object using pickle
    with open(Parameters.PATH + 'history.pkl', 'wb') as file:
        pickle.dump(history, file)
