from keras import *
from keras.applications import ResNet50

from pyspec.machine.model.cnn import CNNClassificationModel


class Resnet50CNNModel(CNNClassificationModel):

    def build(self) -> Model:
        model = ResNet50(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes =2
        )
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()
        return model
