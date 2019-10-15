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

    def generate_validation_generator(self, validate_df: DataFrame, generator: LabelGenerator):
        return super().generate_validation_generator(validate_df, generator)

    def generate_training_generator(self, train_df: DataFrame, generator: LabelGenerator):
        return super().generate_training_generator(train_df, generator)


class SimilarityModel(MultiInputCNNModel):
    """
    computes the similarity between 2 spectra records and is a base class. Concrete implemntations will define the actual models, etc
    """

    def build(self) -> Model:
        pass

    def predict(self, first: Spectra, second: Spectra, encode: Encoder) -> float:
        """
        predicts a similarity score between 2 different spectra, with the given encode.
        score is between 0 and 1. 0 for none identical at all, 1 for identical match
        :param first:
        :param second:
        :param encode:
        :return:
        """

        pass

    def generate_validation_generator(self, validate_df: DataFrame, generator: LabelGenerator):
        return super().generate_validation_generator(validate_df, generator)

    def generate_training_generator(self, train_df: DataFrame, generator: LabelGenerator):
        return super().generate_training_generator(train_df, generator)
