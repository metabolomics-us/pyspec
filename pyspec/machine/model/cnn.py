import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from keras import Model

from pandas import DataFrame
from sklearn.model_selection import train_test_split
from typing import Tuple, List

from pyspec.machine.labels.generate_labels import LabelGenerator


class CNNClassificationModel(ABC):
    """
    provides us with a simple classification model
    """

    def __str__(self) -> str:
        return self.__class__.__name__

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
        self.seed = np.random.seed()

    @abstractmethod
    def build(self) -> Model:
        """
        builds the internal keras model
        :return:
        """

    def train(self, input: str, generator: LabelGenerator, test_size=0.20, epochs=5):
        """
        trains a model for us, based on the input
        :param input:
        :return:
        """
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from keras_preprocessing.image import ImageDataGenerator

        earlystop = EarlyStopping(patience=10)
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                    patience=2,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=0.00001)
        callbacks = [earlystop, learning_rate_reduction]

        dataframe = generator.generate_dataframe(input)

        assert dataframe['file'].apply(lambda x: os.path.exists(x)).all(), 'please ensure all files exist!'

        train_df, validate_df = train_test_split(dataframe, test_size=test_size, random_state=42)
        train_df = train_df.reset_index(drop=True)
        validate_df = validate_df.reset_index(drop=True)

        if self.plots:
            train_df['class'].value_counts().plot.bar()
            plt.title("training classes {}".format(self.get_name()))
            plt.show()

            validate_df['class'].value_counts().plot.bar()
            plt.title("validations classes {}".format(self.get_name()))
            plt.show()

        total_train = train_df.shape[0]
        total_validate = validate_df.shape[0]

        train_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            None,
            x_col='file',
            y_col='class',
            target_size=(self.width, self.height),
            class_mode='categorical',
            batch_size=self.batch_size
        )

        validation_datagen = ImageDataGenerator()
        validation_generator = validation_datagen.flow_from_dataframe(
            validate_df,
            None,
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
            validation_steps=total_validate / self.batch_size,
            steps_per_epoch=total_train / self.batch_size,
            callbacks=callbacks
        )

        if self.plots:
            self.plot_training(epochs, history)

        model.save_weights("{}/{}_model.h5".format(input, self.get_name()))
        model = None

        from keras import backend as K
        K.clear_session()

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

        plt.title("training report {}".format(self.get_name()))
        plt.show()

    def predict_from_dataframe(self, input: str, dataframe: DataFrame, file_column: str = "file",
                               class_column: str = "class") -> DataFrame:
        """
        predicts from an incomming dataframe and a specified colum and returns
        a dataframe with the predictions
        :param input: folder where the trained model is located
        :param dataframe:
        :param file_column:
        :return:
        """
        m = self.build()
        m.load_weights("{}/{}_model.h5".format(input, self.get_name()))

        from keras_preprocessing.image import ImageDataGenerator
        test_gen = ImageDataGenerator()

        nb_samples = dataframe.shape[0]
        test_generator = test_gen.flow_from_dataframe(
            dataframe,
            directory=None,
            x_col=file_column,
            y_col=None,
            class_mode=None,
            target_size=(self.width, self.height),
            batch_size=self.batch_size,
            shuffle=False
        )

        predict = m.predict_generator(test_generator, steps=np.ceil(nb_samples / self.batch_size))

        assert len(predict) > 0, "sorry we were not able to predict anything!"
        dataframe[class_column] = np.argmax(predict, axis=-1)

        if self.plots:
            dataframe[class_column].value_counts().plot.bar()
            plt.show()

        return dataframe

    def predict_from_files(self, input: str, files: List[str]) -> List[Tuple[str, str]]:
        """
        predicts from a list of files and returns a list of tuples
        :param input: folder where the model is located
        :param files: list of absolute file names you would like to load and predict

        :return:
        """
        data = []
        for x in files:
            data.append(
                {
                    'file': os.path.abspath(x)
                }
            )

        dataframe = self.predict_from_dataframe(input=input, dataframe=DataFrame(data))
        return list(
            dataframe.itertuples(index=False, name=None)
        )

    def predict_from_file(self, input: str, file: str) -> Tuple[str, str]:
        """

        this does the prediction based on the given file for us
        :param input: the location where the model is located as directory
        :param file: the file name you want to test
        :return:
        """

        return self.predict_from_files(input, [file])[0]

    def predict_from_directory(self, input: str, dict: str, callback):
        """
        predicts from a dictionary
        :param input:
        :param dict:
        :param callback:
        :return:
        """
        m = self.build()
        m.load_weights("{}/{}_model.h5".format(input, self.get_name()))

        from keras_preprocessing.image import ImageDataGenerator
        test_gen = ImageDataGenerator()

        for file in os.listdir(dict):
            f = os.path.abspath("{}/{}".format(dict, file))
            if os.path.isfile(f):
                dataframe = DataFrame([{'file': f}])

                nb_samples = dataframe.shape[0]
                test_generator = test_gen.flow_from_dataframe(
                    dataframe,
                    directory=None,
                    x_col="file",
                    y_col=None,
                    class_mode=None,
                    target_size=(self.width, self.height),
                    batch_size=self.batch_size,
                    shuffle=False
                )

                predict = m.predict_generator(test_generator, steps=np.ceil(nb_samples / self.batch_size))
                cat = np.argmax(predict, axis=-1)[0]
                callback(file, cat)

    def get_name(self) -> str:
        """
        returns the name of this model, by default this is the concrete class name
        :return:
        """
        return self.__class__.__name__
