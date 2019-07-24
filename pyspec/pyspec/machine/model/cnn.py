import os
import warnings
from abc import ABC, abstractmethod

import h5py
import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from keras.backend import set_session
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils import multi_gpu_model
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from typing import Tuple, List
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_preprocessing.image import ImageDataGenerator
from pyspec.loader import Spectra
from pyspec.machine.labels.generate_labels import LabelGenerator
from pyspec.machine.spectra import Encoder


class MultiGPUModelCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        0erbose: verbosity mode, 0 or 1.
        0ave_best_only: if `save_best_only=True`,
            0he latest best model according to
            0he quantity monitored will not be overwritten.
        0ode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MultiGPUModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.layers[-2].save_weights(filepath, overwrite=True)
                        else:
                            self.model.layers[-2].save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.layers[-2].save_weights(filepath, overwrite=True)
                else:
                    self.model.layers[-2].save(filepath, overwrite=True)


class CNNClassificationModel(ABC):
    """
    provides us with a simple classification model
    """

    def __str__(self) -> str:
        return self.__class__.__name__

    def __init__(self, width: int, height: int, channels: int, plots: bool = False, batch_size=15, seed=12345):
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

    def fix_seed(self):
        """
        fixes the random seed so results are repeatable
        :return:
        """
        from numpy.random import seed
        seed(self.seed)
        from tensorflow import set_random_seed
        set_random_seed(self.seed)

    @abstractmethod
    def build(self) -> Model:
        """
        builds the internal keras model
        :return:
        """

    def configure_session(self):
        """
        configures the tensorflow session for us
        :return:
        """

        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        #        config.log_device_placement = True  # to log device placement (on which device the operation ran)
        sess = tf.Session(config=config)
        return sess

    def train(self, input: str, generator: LabelGenerator, test_size=0.20, epochs=5, gpus=1, verbose=1):
        """
        trains a model for us, based on the input
        :param input:
        :return:
        """

        self.fix_seed()
        earlystop = EarlyStopping(patience=10)
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                    patience=2,
                                                    verbose=verbose,
                                                    factor=0.5,
                                                    min_lr=0.00001)
        callbacks = [earlystop, learning_rate_reduction]

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
        dataframe = generator.generate_dataframe(input)

        assert dataframe['file'].apply(lambda x: os.path.exists(x)).all(), 'please ensure all files exist!'

        train_df, validate_df = train_test_split(dataframe, test_size=test_size, random_state=42)
        train_df = train_df.reset_index(drop=True)
        validate_df = validate_df.reset_index(drop=True)

        #       if self.plots:
        #           train_df['class'].value_counts().plot.bar()
        #           plt.title("training classes {}".format(self.get_name()))
        #           plt.show()

        #           validate_df['class'].value_counts().plot.bar()
        #           plt.title("validations classes {}".format(self.get_name()))
        #           plt.show()

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
            batch_size=self.batch_size,
        )

        validation_datagen = ImageDataGenerator()
        validation_generator = validation_datagen.flow_from_dataframe(
            validate_df,
            None,
            x_col='file',
            y_col='class',
            target_size=(self.width, self.height),
            class_mode='categorical',
            batch_size=self.batch_size,
        )

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
            verbose=verbose
        )

        if self.plots:
            self.plot_training(epochs, history)

        del model
        from keras import backend as K
        K.clear_session()

    def get_model_file(self, input):
        return "{}/{}_model.h5".format(input, self.get_name())

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

        plt.title("training report {}, bs = {}".format(self.get_name(), self.batch_size))
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

    def get_model(self, input):
        m = self.build()
        m.load_weights(self.get_model_file(input))
        return m

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

    def get_name(self) -> str:

        """
        returns the name of this model, by default this is the concrete class name
        :return:
        """
        return "{}_bs_{}".format(self.__class__.__name__, self.batch_size)
