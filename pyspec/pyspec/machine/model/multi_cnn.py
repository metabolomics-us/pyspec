from keras import Model, Input
from keras.applications import ResNet50
from keras.layers import Dense, concatenate, Conv2D, MaxPooling2D, Flatten
from keras.utils import plot_model

from pyspec.loader import Spectra
from pyspec.machine.model.cnn import MultiInputCNNModel
from pyspec.machine.spectra import Encoder


class SimilarityModel(MultiInputCNNModel):
    """
    computes the similarity between 2 spectra records and is a base class. Concrete implemntations will define the actual models, etc
    """

    def build(self) -> Model:
        first_spectra = Input(shape=(self.width, self.height, self.channels))
        second_spectra = Input(shape=(self.width, self.height, self.channels))

        x = self.rename_layers(self.create_cnn(first_spectra), "library")
        y = self.rename_layers(self.create_cnn(second_spectra), "unknown")

        # combine the output of the two branches
        combined = concatenate([x.output, y.output])

        # output layer
        z = Dense(2, activation="softmax")(combined)

        # our model will accept the inputs of the two branches and
        # then output a single value
        model = Model(inputs=[x.input, y.input], outputs=z)

        plot_model(model, to_file='model_plot-{}.png'.format(self.__class__.__name__), show_shapes=True,
                   show_layer_names=True)
        return model

    def create_cnn(self, visible):
        """
        creates the CNN model for the image recognition. This one is a useless dummy and should be subclassed with an appropriate model
        :param visible:
        :return:
        """
        conv1 = Conv2D(32, kernel_size=4, activation='relu')(visible)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        flat = Flatten()(pool2)
        hidden1 = Dense(10, activation='relu')(flat)
        output = Dense(1, activation='sigmoid')(hidden1)

        model = Model(inputs=visible, outputs=output)

        return model

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

    def rename_layers(self, model, identifier):
        """
        renames all layers in this model to ensure uniqueness
        :param model: 
        :param identifier: 
        :return: 
        """
        for layer in model.layers:
            layer.name = "{}_{}".format(layer.name, identifier)

        return model


class Resnet50SimilarityModel(SimilarityModel):
    def create_cnn(self, visible):
        base = ResNet50(include_top=False, input_tensor=visible)
        x = Flatten()(base.output)
        x = Dense(2, activation='softmax')(x)
        model = Model(inputs=base.inputs, outputs=x)
        return model
