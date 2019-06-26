from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization


class ClassificationModel:
    """
    provides us with a simple classification model
    """

    def __init__(self, width: int, height: int, channels: int):
        """
        defines the model size
        :param width:
        :param height:
        :param channels:
        """

        self.width = width
        self.height = height
        self.channels = channels

    def build(self):
        """
        builds the internal keras model
        :return:
        """

        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.width, self.height, self.channels)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))  # 2 because we have cat and dog classes

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        model.summary()

    def train(self):
        pass

    def predict(self):
        pass
