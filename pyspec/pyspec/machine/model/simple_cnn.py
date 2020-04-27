from keras import Model, Input

from pyspec.machine.model.cnn import SingleInputCNNModel
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.models import Sequential


class SimpleCNNModel(SingleInputCNNModel):
    """
    very basic cnn model to compare 2 images
    """

    def build(self) -> Model:
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


        return model


class PoolingCNNModel(SingleInputCNNModel):
    """
    utilize pooling resources
    """

    def build(self) -> Model:
        model = Sequential()

        # Step 1
        model.add(Conv2D(filters=32, kernel_size=(3, 3),
                         padding='same', activation='relu', input_shape=(self.width, self.height, self.channels)))

        # Step 2 - Pooling
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Step 1
        model.add(Conv2D(filters=48, kernel_size=(3, 3), padding='same',
                         activation='relu'))
        # Step 2 - Pooling
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Step 1
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same',
                         activation='relu'))
        # Step 2 - Pooling
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Step 1
        model.add(Conv2D(filters=96, kernel_size=(3, 3), padding='same',
                         activation='relu'))
        # Step 2 - Pooling
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Step 3 - Flattening
        model.add(Flatten())

        # Step 4 - Full connection

        model.add(Dense(256, activation='relu'))
        # Dropout
        model.add(Dropout(0.5))

        model.add(Dense(2, activation='softmax'))

        return model
