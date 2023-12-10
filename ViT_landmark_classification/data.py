from tensorflow import keras
from pathlib import Path
from tqdm import tqdm
import numpy as np
from utils import Config


class Data:
    states = {i: state.name for i, state in enumerate(Path(f'{Config.PATH}data').glob('*/'))}

    def __init__(self,
                 validation_split: float = 0.15,
                 image_size: tuple = (256, 256),
                 batch_size: int = 32,
                 seed: int = 123,
                 label_mode: str = 'categorical'):

        self.val_split = validation_split
        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed
        self.label_mode = label_mode

    def load_train(self,
                   name: str,
                   path: str = Config.PATH):
        return keras.preprocessing.image_dataset_from_directory(
            f'{path}{name}',
            validation_split=self.val_split,
            subset="training",
            seed=self.seed,
            label_mode=self.label_mode,
            image_size=self.image_size,
            batch_size=self.batch_size)

    def load_val(self,
                 name: str,
                 path: str = Config.PATH):
        return keras.preprocessing.image_dataset_from_directory(
            f'{path}{name}',
            validation_split=self.val_split,
            subset="validation",
            seed=self.seed,
            label_mode=self.label_mode,
            image_size=self.image_size,
            batch_size=self.batch_size)

    def load_test(self,
                  path: str = Config.PATH) -> tuple:
        x_test, y_test = [], []
        test_ds = keras.preprocessing.image_dataset_from_directory(
            f'{path}test_data',
            label_mode=self.label_mode,
            image_size=self.image_size,
            batch_size=self.batch_size)

        for images, labels in tqdm(test_ds.unbatch()):
            x_test.append(images.numpy())
            y_test.append(labels.numpy())

        return np.array(x_test), np.array(y_test)