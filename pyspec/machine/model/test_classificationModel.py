import pytest

from pyspec.machine.labels.generate_labels import DirectoryLabelGenerator
from pyspec.machine.model.Xception import XceptionModel
from pyspec.machine.model.resnet50 import Resnet50CNNModel
from pyspec.machine.model.simple_cnn import SimpleCNNModel, PoolingCNNModel

batchsize = 2
epochs = 50

models = [
    XceptionModel(width=500, height=500, channels=3, plots=True, batch_size=batchsize),
    Resnet50CNNModel(width=500, height=500, channels=3, plots=True, batch_size=batchsize),
    PoolingCNNModel(width=500, height=500, channels=3, plots=True, batch_size=batchsize),
    SimpleCNNModel(width=500, height=500, channels=3, plots=True, batch_size=batchsize)
]
datasets = ['clean_dirty', 'pos_neg']


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dataset", datasets)
def test_train(model, dataset):
    """
    tests the training, which also generates a model file, whihc is later used for other tests.
    :return:
    """

    generator = DirectoryLabelGenerator()
    try:
        result = model.train(input="datasets/{}".format(dataset), generator=generator, epochs=epochs)
    finally:
        del model


@pytest.mark.parametrize("dataset", datasets)
@pytest.mark.parametrize("model", models)
def test_predict_from_dataframe(model, dataset):
    generator = DirectoryLabelGenerator()
    test_df = generator.generate_test_dataframe("datasets/{}".format(dataset), abs=True)
    result = model.predict_from_dataframe(dataframe=test_df, input="datasets/{}".format(dataset))
    print(result)


@pytest.mark.parametrize("model", models)
def test_predict_from_file(model):
    result = model.predict_from_file(
        file="datasets/clean_dirty/test/dirty/splash10-0a4i-8976455223-5124e0bff8404b3090d3.png",
        input="datasets/clean_dirty")
    print(result)
