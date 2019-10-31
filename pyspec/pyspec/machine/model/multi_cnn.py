from abc import abstractmethod

from keras import Model, Input
from keras.applications import ResNet50
from keras.layers import Dense, concatenate, Conv2D, MaxPooling2D, Flatten
from keras.utils import plot_model

from pyspec.loader import Spectra
from pyspec.machine.labels.similarity_labels import SimilarityTuple, SimilarityDatasetLabelGenerator, \
    EnhancedSimilarityDatasetLabelGenerator
from pyspec.machine.model.cnn import MultiInputCNNModel
from pyspec.machine.spectra import Encoder
import numpy as np


class SimilarityModel(MultiInputCNNModel):
    """
    computes the similarity between 2 spectra records and is a base class. Concrete implemntations will define the actual models, etc
    """

    def build(self) -> Model:
        first_spectra = Input(shape=(self.width, self.height, self.channels))
        second_spectra = Input(shape=(self.width, self.height, self.channels))
        similarity_measures = Input(shape=(len(list(SimilarityTuple())),))
        x = self.rename_layers(self.create_cnn(first_spectra), "library")
        y = self.rename_layers(self.create_cnn(second_spectra), "unknown")
        xy = self.rename_layers(self.create_similarity_measures_nn(similarity_measures), "measures")

        # combine the output of the two branches
        combined = concatenate([x.output, y.output, xy.output])

        # output layer
        z = Dense(2, activation="softmax")(combined)

        # our model will accept the inputs of the different  branches and
        # then output a single value
        model = Model(inputs=[x.input, y.input, xy.input], outputs=z)

        plot_model(model, to_file='model_plot-{}.png'.format(self.__class__.__name__), show_shapes=True,
                   show_layer_names=True)
        return model

    @abstractmethod
    def create_similarity_measures_nn(self, input) -> Model:
        """
        generates a model to evaluate the similarity measures in a neural network
        :param input:
        :return:
        """

    @abstractmethod
    def create_cnn(self, visible) -> Model:
        """
        creates the CNN model for the image recognition. This one is a useless dummy and should be subclassed with an appropriate model
        :param visible:
        :return:
        """

    def predict(self, input: str, first: Spectra, second: Spectra, encode: Encoder) -> float:
        """
        predicts a similarity score between 2 different spectra, with the given encode.
        score is between 0 and 1. 0 for none identical at all, 1 for identical match
        :param first:
        :param second:
        :param encode:
        :return:
        """

        model = self.get_model(input)
        encode.height = self.height
        encode.width = self.width

        encoded_1 = np.expand_dims(
            np.fromstring(encode.encode(first), dtype='uint8').reshape((self.width, self.height, self.channels)),
            axis=0)
        encoded_2 = np.expand_dims(
            np.fromstring(encode.encode(second), dtype='uint8').reshape((self.width, self.height, self.channels)),
            axis=0)

        measures = EnhancedSimilarityDatasetLabelGenerator.compute_similarities(first, second)

        request = [encoded_1, encoded_2, measures]

        prediction = model.predict(request, batch_size=self.batch_size)
        # assemble to numpy array
        return prediction


class Resnet50SimilarityModel(SimilarityModel):
    """
    defines a simple resnet 50 based architecture
    """

    def create_similarity_measures_nn(self, input) -> Model:
        x = Dense(8, activation="relu")(input)
        x = Dense(4, activation="relu")(x)
        x = Model(inputs=input, outputs=x)
        return x

    def create_cnn(self, visible):
        base = ResNet50(include_top=False, input_tensor=visible)
        x = Flatten()(base.output)
        x = Dense(2, activation='softmax')(x)
        model = Model(inputs=base.inputs, outputs=x)
        return model
