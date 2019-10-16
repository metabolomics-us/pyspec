from typing import List, Any, Tuple

import keras
import numpy as np
from keras.utils import Sequence

from pyspec.loader import Spectra
from pyspec.machine.spectra import Encoder


class SpectraDataGenerator(Sequence):
    """
    generates spectra for us from an input dataframe
    """

    def __getitem__(self, index):
        indexes = self.spectra[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def __len__(self):
        return int(np.floor(len(self.spectra) / self.batch_size))

    def __init__(self, spectra: List[Tuple[Spectra, str]], encoder: Encoder, n_classes: int = 2,
                 batch_size: int = 32):
        """
        generate numpy arrays for the given spectra
        :param spectra:
        :param batch_size:
        :param encoder:
        """
        self.spectra = spectra
        self.batch_size = batch_size
        self.encoder = encoder
        self.n_classes = n_classes
        self.image_shape = (encoder.height, encoder.width) + (3,)

        # build labels, based on index

        self.labels = list(set(map(lambda x: x[1], spectra)))
        self.class_indices = dict(zip(self.labels, range(len(self.labels))))

    def __data_generation(self, spectra: List[Tuple[Spectra, str]]):
        """
        generates the actual numpy presentation of our spectra as image in an numpy array
        :param spectra:
        :return:
        """
        X = np.zeros((len(spectra),) + self.image_shape)
        y = np.empty(self.batch_size)

        # Generate data
        for i, spec in enumerate(spectra):
            # Store sample
            try:
                encoded = self.encoder.encode(spec[0])
                image = np.fromstring(encoded, dtype='uint8')
                image = image.reshape((self.encoder.width, self.encoder.height, 3))
                image = np.expand_dims(image, axis=0)
                X[i] = image

                # Store class
                label = self.class_indices[spec[1]]
                y[i] = label
            except Exception as e:
                pass

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
