import numpy as np
from keras_preprocessing.image import ImageDataGenerator, load_img

from pyspec.machine.labels.generate_labels import DirectoryLabelGenerator
from pyspec.machine.model.cnn import ClassificationModel

import matplotlib.pyplot as plt

batchsize = 2
epochs = 50


def test_train():
    """
    tests the training, which also generates a model file, whihc is later used for other tests.
    :return:
    """

    model = ClassificationModel(width=500, height=500, channels=3, plots=True, batch_size=batchsize)
    generator = DirectoryLabelGenerator()

    test_df = generator.generate_test_dataframe("datasets/clean_dirty")

    m = model.train("datasets/clean_dirty", generator, epochs=epochs)


def test_predict_from_dataframe():
    model = ClassificationModel(width=500, height=500, channels=3, plots=True, batch_size=batchsize)
    generator = DirectoryLabelGenerator()
    test_df = generator.generate_test_dataframe("datasets/clean_dirty", abs=True)
    result = model.predict_from_dataframe(dataframe=test_df, input="datasets/clean_dirty")
    print(result)


def test_predict_from_file():
    model = ClassificationModel(width=500, height=500, channels=3, plots=True, batch_size=batchsize)
    result = model.predict_from_file(file="datasets/clean_dirty/test/dirty/splash10-0a4i-8976455223-5124e0bff8404b3090d3.png", input="datasets/clean_dirty")
    print(result)

def test_predict():
    model = ClassificationModel(width=500, height=500, channels=3, plots=True, batch_size=batchsize)
    generator = DirectoryLabelGenerator()
    m = model.build()
    m.load_weights("datasets/clean_dirty/model.h5")

    test_gen = ImageDataGenerator()

    test_df = generator.generate_test_dataframe("datasets/clean_dirty", abs=False)

    nb_samples = test_df.shape[0]
    test_generator = test_gen.flow_from_dataframe(
        test_df,
        "datasets/clean_dirty",
        x_col='file',
        y_col=None,
        class_mode=None,
        target_size=(500, 500),
        batch_size=batchsize,
        shuffle=False
    )

    predict = m.predict_generator(test_generator, steps=np.ceil(nb_samples / batchsize))

    test_df['class'] = np.argmax(predict, axis=-1)
    test_df['class'].value_counts().plot.bar()
    print(test_df)
    plt.show()

    sample_test = test_df
    plt.figure(figsize=(12, 24))
    for index, row in sample_test.iterrows():
        filename = row['file']
        category = row['class']
        img = load_img("datasets/clean_dirty/" + filename, target_size=(500, 500))
        plt.subplot(12, 2, index + 1)
        plt.imshow(img)
        plt.title(filename + '\n(' + "{}".format(category) + ')')
    plt.tight_layout()
    plt.show()
