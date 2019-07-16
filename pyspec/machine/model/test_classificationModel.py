import importlib

import pytest
from numba import cuda

from pyspec.machine.labels.generate_labels import DirectoryLabelGenerator

batchsize = 2
epochs = 5

models = [
    'XceptionModel',
    'Resnet50CNNModel',
    'VGG16Model',
    'VGG19Model',
    'InceptionModel',
    'InceptionResNetModel',
    'MobileNetModel',
    'MobileNetV2Model',
    'DenseNet121Model',
    'DenseNet169Model',
    'DenseNet201Model',
    'NASNetMobileModel',
    'NASNetLargeModel'
]
datasets = ['clean_dirty']  # , 'pos_neg']


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dataset", datasets)
def test_train(model, dataset):
    """
    tests the training, which also generates a model file, whihc is later used for other tests.
    :return:
    """
    model = init_model(model)

    generator = DirectoryLabelGenerator()
    try:
        result = model.train(input="datasets/{}".format(dataset), generator=generator, epochs=epochs)
    finally:
        del model
        from keras import backend as K
        K.clear_session()


def init_model(model):
    class_ = getattr(importlib.import_module("pyspec.machine.model.application"), model)
    model = class_(width=500, height=500, channels=3, plots=True, batch_size=batchsize)
    return model


@pytest.mark.parametrize("dataset", datasets)
@pytest.mark.parametrize("model", models)
def test_predict_from_dataframe(model, dataset):
    model = init_model(model)
    generator = DirectoryLabelGenerator()
    test_df = generator.generate_test_dataframe("datasets/{}".format(dataset), abs=True)
    result = model.predict_from_dataframe(dataframe=test_df, input="datasets/{}".format(dataset))
    print(result)


@pytest.mark.parametrize("model", models)
def test_predict_from_file(model):
    model = init_model(model)
    result = model.predict_from_file(
        file="datasets/clean_dirty/test/dirty/splash10-0a4i-8976455223-5124e0bff8404b3090d3.png",
        input="datasets/clean_dirty")
    print(result)
