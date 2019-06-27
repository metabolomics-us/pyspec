from unittest import TestCase

from keras_preprocessing.image import ImageDataGenerator

from pyspec.machine.labels.generate_labels import DirectoryLabelGenerator
from pyspec.machine.model.cnn import ClassificationModel
import numpy as np


def test_train():
    model = ClassificationModel(width=500, height=500, channels=3, plots=True, batch_size=15)
    generator = DirectoryLabelGenerator()

    test_df = generator.generate_test_dataframe("datasets/clean_dirty")

    m = model.train("datasets/clean_dirty", generator)

    m.save_weights("datasets/clean_dirty/model.h5")

    nb_samples = test_df.shape[0]

    test_gen = ImageDataGenerator()
    test_generator = test_gen.flow_from_dataframe(
        test_df,
        "datasets/clean_dirty",
        x_col='file',
        y_col=None,
        class_mode=None,
        target_size=(500, 500),
        batch_size=15,
        shuffle=False
    )

    predict = m.predict_generator(test_generator, steps=np.ceil(nb_samples / 15))
