from keras import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt

from pyspec.machine.labels.generate_labels import LabelGenerator


class ClassificationModel:
    """
    provides us with a simple classification model
    """

    def __init__(self, width: int, height: int, channels: int, plots: bool = False, batch_size=15):
        """
        defines the model size
        :param width:
        :param height:
        :param channels:
        """

        self.width = width
        self.height = height
        self.channels = channels
        self.plots = plots
        self.batch_size = batch_size

    def build(self) -> Model:
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

        return model

    def train(self, input: str, generator: LabelGenerator, test_size=0.20, epochs=5) -> Model:
        """
        trains a model for us, based on the input
        :param input:
        :return:
        """
        earlystop = EarlyStopping(patience=10)
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                    patience=2,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=0.00001)
        callbacks = [earlystop, learning_rate_reduction]

        dataframe = generator.generate_dataframe(input)

        train_df, validate_df = train_test_split(dataframe, test_size=test_size, random_state=42)
        train_df = train_df.reset_index(drop=True)
        validate_df = validate_df.reset_index(drop=True)

        if self.plots:
            train_df['class'].value_counts().plot.bar()
            plt.show()

            validate_df['class'].value_counts().plot.bar()
            plt.show()

        total_train = train_df.shape[0]
        total_validate = validate_df.shape[0]

        train_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            input,
            x_col='file',
            y_col='class',
            target_size=(self.width, self.height),
            class_mode='categorical',
            batch_size=self.batch_size
        )

        validation_datagen = ImageDataGenerator()
        validation_generator = validation_datagen.flow_from_dataframe(
            validate_df,
            input,
            x_col='file',
            y_col='class',
            target_size=(self.width, self.height),
            class_mode='categorical',
            batch_size=self.batch_size
        )

        model = self.build()

        history = model.fit_generator(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=total_validate // self.batch_size,
            steps_per_epoch=total_train // self.batch_size,
            callbacks=callbacks
        )

        if self.plots:
            self.plot_training(epochs, history)

        return model

    def plot_training(self, epochs, history):
        """
        plots the training statistics for us
        :param epochs:
        :param history:
        :return:
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        ax1.plot(history.history['loss'], color='b', label="Training loss")
        ax1.plot(history.history['val_loss'], color='r', label="validation loss")
        ax1.set_xticks(np.arange(1, epochs, 1))
        ax1.set_yticks(np.arange(0, 1, 0.1))
        ax2.plot(history.history['acc'], color='b', label="Training accuracy")
        ax2.plot(history.history['val_acc'], color='r', label="Validation accuracy")
        ax2.set_xticks(np.arange(1, epochs, 1))
        legend = plt.legend(loc='best', shadow=True)
        plt.tight_layout()
        plt.show()

    def predict(self, image) -> str:
        """
        this predicts the class of the given image for us
        :param image:
        :return:
        """
        pass
