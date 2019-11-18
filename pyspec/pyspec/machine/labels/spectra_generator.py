import traceback
from collections import OrderedDict
from typing import List, Any, Tuple

import keras
import numpy as np
from keras.utils import Sequence

from pyspec.loader import Spectra
from pyspec.machine.spectra import Encoder


class SimilarityMeasureGenerator(Sequence):

    def __init__(self, data: List, batch_size: int = 32, n_classes=2):
        """

        :param data:
        :param batch_size:
        """
        self.data = data
        self.batch_size = batch_size
        self.shape = (data[0].compute_size(),)

        self.labels = list(set(map(lambda x: x[1], data)))
        self.class_indices = OrderedDict(dict(zip(self.labels, range(len(self.labels)))))

        self.n_classes = n_classes

    def __getitem__(self, index):
        indexes = self.data[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def __data_generation(self, data):
        X = np.zeros((len(data),) + self.shape)
        y = np.empty(self.batch_size)

        # Generate data
        for i, value in enumerate(data):
            try:

                X[i] = value[0].to_nd()  # np.array(list(value[0]))

                # Store class
                label = self.class_indices[value[1]]
                y[i] = label
            except Exception as e:
                pass

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))


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
        self.class_indices = OrderedDict(dict(zip(self.labels, range(len(self.labels)))))

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
                image_rgb = image.reshape((self.encoder.width, self.encoder.height, 3))
                image = np.expand_dims(image_rgb, axis=0)
                X[i] = image

                # Store class
                label = self.class_indices[spec[1]]
                y[i] = label

            #               just plotting all the spectra which are used for training
            #               try:
            #                   import matplotlib.pyplot as plt

            #                   fig2 = plt.figure()
            #                   ax2 = fig2.add_subplot(1, 1, 1)
            #                   ax2.imshow(image_rgb)
            #                   plt.show()
            #               except Exception as e:
            #                   traceback.print_exc()

            except Exception as e:
                pass

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
