from keras.applications import Xception
from keras.models import Model

from pyspec.machine.model.cnn import CNNClassificationModel


class XceptionModel(CNNClassificationModel):
    """
    keras XCEPTION model
    """

    def build(self) -> Model:
        model = Xception(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes = 2
        )
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        return model
