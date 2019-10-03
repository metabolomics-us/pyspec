from keras import Model
from pandas import DataFrame

from pyspec.loader import Spectra
from pyspec.machine.labels.generate_labels import LabelGenerator
from pyspec.machine.model.cnn import MultiInputCNNModel
from pyspec.machine.spectra import Encoder
import numpy as np


class SimpleMultiCNNModel(MultiInputCNNModel):
    """
    a model utilizing multiple inputs to distinguish clean and dirty spectra
    """

    def build(self) -> Model:
        pass

    def generate_validation_generator(self, validate_df: DataFrame, generator:LabelGenerator):
        return super().generate_validation_generator(validate_df,generator)

    def generate_training_generator(self, train_df: DataFrame, generator:LabelGenerator):
        return super().generate_training_generator(train_df,generator)

    def predict_from_spectra(self, input: str, spectra: Spectra, encoder: Encoder) -> str:
        """
        predicts the class from the given spectra
        :param spectra:
        :return:
        """
        model = self.get_model(input)
        spectra = encoder.encode(spectra)

        # convert the incoming data to a numpy array wxhxc
        data = np.fromstring(spectra, dtype='uint8').reshape((self.width, self.height, self.channels))
        # expand it by 1 dimension
        data = np.expand_dims(data, axis=0)

        y_proba = model.predict(data, batch_size=self.batch_size)
        y_classes = y_proba.argmax(axis=-1)
        return y_classes[0]
