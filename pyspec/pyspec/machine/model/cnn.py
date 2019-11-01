import multiprocessing
import os
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from keras.backend import set_session
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import multi_gpu_model
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from pyspec.loader import Spectra
from pyspec.machine.labels.generate_labels import LabelGenerator
from pyspec.machine.spectra import Encoder
from pyspec.machine.util.checkpoint import MultiGPUModelCheckpoint
from pyspec.machine.util.gpu import get_gpu_count


class CNNClassificationModel(ABC):
    """
    provides us with a simple classification model
    """

    def __str__(self) -> str:
        return self.__class__.__name__

    def __init__(self, width: int, height: int, channels: int, plots: bool = False, batch_size=15, seed=12345,
                 early_stop=False, tensor_board=True, workers: Optional[int] = None):
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
        self.seed = seed
        self.early_stop = early_stop
        self.tensor_board = tensor_board

        if workers is None:
            workers = multiprocessing.cpu_count() - 2

            if workers < 1:
                workers = 1

        self.workers = workers

    @abstractmethod
    def build(self) -> Model:
        """
        builds the internal keras model
        :return:
        """

    def fix_seed(self):
        """
        fixes the random seed so results are repeatable
        :return:
        """
        from numpy.random import seed
        seed(self.seed)
        from tensorflow import set_random_seed
        set_random_seed(self.seed)

    def configure_session(self):
        """
        configures the tensorflow session for us
        :return:
        """
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        #        config.log_device_placement = True  # to log device placement (on which device the operation ran)
        sess = tf.Session(config=config)
        return sess

    def train(self, input: str, generator: LabelGenerator, encoder: Encoder, test_size: Optional[float] = 0.20,
              epochs=5, gpus=None,
              verbose=1):
        """

        :param input: location to the input for the label generator
        :param generator: which label generator to use
        :param encoder: an encoder how to encode the loaded data
        :param test_size: None, if label provider provides test/training data, otherwise a float between 0-1
        :param epochs: how many epochs you would like to train for
        :param gpus: None to use all gpus, otherwise a number
        :param verbose: do you want verbose logging
        :return:
        """
        if gpus is None:
            gpus = get_gpu_count()

        assert encoder is not None, "please ensure you provide an encoder!"
        encoder.width = self.width
        encoder.height = self.height

        self.fix_seed()
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                    patience=2,
                                                    verbose=verbose,
                                                    factor=0.5,
                                                    min_lr=0.00001)
        callbacks = [learning_rate_reduction]

        self.configure_checkpoints(callbacks, gpus, input, verbose)
        train_df, validate_df = self.generate_dataset(generator, input, test_size)
        train_df = train_df.reset_index(drop=True)
        validate_df = validate_df.reset_index(drop=True)

        total_train = train_df.shape[0]
        total_validate = validate_df.shape[0]

        train_generator = generator.get_data_generator(train_df, width=self.width, height=self.height,
                                                       batch_size=self.batch_size, encoder=encoder)
        validation_generator = generator.get_data_generator(validate_df, width=self.width, height=self.height,
                                                            batch_size=self.batch_size, encoder=encoder)

        set_session(self.configure_session())
        model = self.build()

        # allow to use multiple gpus if available
        if gpus > 1:
            print("using multi gpu mode!")
            model = multi_gpu_model(model, gpus=gpus, cpu_relocation=True)
        else:
            print("using single GPU mode!")

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        history = model.fit_generator(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=total_validate / self.batch_size,
            steps_per_epoch=total_train / self.batch_size,
            callbacks=callbacks,
            use_multiprocessing=True,
            verbose=verbose,
            workers=self.workers
        )

        if self.plots:
            self.plot_training(epochs, history, input)

        del model
        from keras import backend as K
        K.clear_session()

    def generate_dataset(self, generator: LabelGenerator, input: str, test_size: Optional[float] = None):
        """
        generates the test and training data for us
        :param generator:
        :param input:
        :param test_size:
        :return:
        """
        dataframe = generator.generate_dataframe(input)

        # use pre defined split of the data
        if test_size is None:
            return dataframe[0], dataframe[1]
        elif generator.contains_test_data() is False:
            if test_size is None:
                # forcing a testsize
                test_size = 0.2
            return train_test_split(dataframe[0], test_size=test_size, random_state=42)
        else:
            # assume we need to split the data
            return train_test_split(dataframe[0], test_size=test_size, random_state=42)

    def configure_checkpoints(self, callbacks, gpus, input, verbose):
        """
        configures the checkpoints for saving the best weights to a model file
        :param callbacks:
        :param gpus:
        :param input:
        :param verbose:
        :return:
        """

        os.makedirs(Path(self.get_model_file(input=input)).parent, exist_ok=True)
        if gpus > 1:
            callbacks.append(
                MultiGPUModelCheckpoint(self.get_model_file(input=input), monitor='val_acc', verbose=verbose,
                                        save_best_only=True,
                                        mode='max')

            )
        else:
            callbacks.append(
                ModelCheckpoint(self.get_model_file(input=input), monitor='val_acc', verbose=verbose,
                                save_best_only=True,
                                mode='max')

            )

        if self.early_stop is True:
            earlystop = EarlyStopping(patience=10)
            callbacks.append(earlystop)

        if self.tensor_board:
            os.makedirs("./tensorboard/logs", exist_ok=True)
            callbacks.append(
                TensorBoard(log_dir='./tensorboard/logs', histogram_freq=0, batch_size=self.batch_size,
                            write_graph=True,
                            write_grads=False, write_images=True, embeddings_freq=0,
                            embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                            update_freq='epoch'))

    def get_model_file(self, input):
        print("loading file in {}".format(input))
        return "{}/{}_model.h5".format(input, self.get_name())

    def plot_training(self, epochs, history, input):
        """
        plots the training statistics for us
        :param epochs:
        :param history:
        :return:
        """

        try:
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

            plt.title("training report {}, bs = {} for {}".format(self.get_name(), self.batch_size, input))
            plt.show()
        except Exception as e:
            # plotting should never kill everything
            traceback.print_stack()
            pass

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
        m = self.get_model(input)

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

        return dataframe

    def get_model(self, input):
        m = self.build()
        m.load_weights(self.get_model_file(input))
        return m

    def get_name(self) -> str:

        """
        returns the name of this model, by default this is the concrete class name
        :return:
        """
        return "{}_bs_{}".format(self.__class__.__name__, self.batch_size)


class SingleInputCNNModel(CNNClassificationModel, ABC):
    """
    works on a single input
    """

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
        m = self.get_model(input)

        from keras_preprocessing.image import ImageDataGenerator
        test_gen = ImageDataGenerator()

        for file in os.listdir(dict):
            f = os.path.abspath("{}/{}".format(dict, file))

            if os.path.isfile(f):
                dataframe = DataFrame([{'file': f}])

                assert os.path.exists(f), "please make sure the file {} exist!".format(f)
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
                callback(file, cat, full_path=f)

    def predict_from_spectra(self, input: str, spectra: Spectra, encoder: Encoder) -> str:
        """
        predicts the class from the given spectra
        :param spectra:
        :return:
        """
        model = self.get_model(input)
        spectra = encoder.encode(spectra)

        # convert the incomming data to a numpy array wxhxc
        data = np.fromstring(spectra, dtype='uint8').reshape((self.width, self.height, self.channels))
        # expand it by 1 dimension
        data = np.expand_dims(data, axis=0)

        y_proba = model.predict(data, batch_size=self.batch_size)
        y_classes = y_proba.argmax(axis=-1)
        return y_classes[0]


class MultiInputCNNModel(CNNClassificationModel, ABC):
    """
    supports multiple inputs
    """

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
