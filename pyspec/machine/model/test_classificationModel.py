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


@pytest.mark.parametrize("model", models)
def test_train(model):
    """
    tests the training, which also generates a model file, whihc is later used for other tests.
    :return:
    """

    generator = DirectoryLabelGenerator()

    test_df = generator.generate_test_dataframe("datasets/clean_dirty")

    m = model.train("datasets/clean_dirty", generator, epochs=epochs)


@pytest.mark.parametrize("model", models)
def test_predict_from_dataframe(model):
    generator = DirectoryLabelGenerator()
    test_df = generator.generate_test_dataframe("datasets/clean_dirty", abs=True)
    result = model.predict_from_dataframe(dataframe=test_df, input="datasets/clean_dirty")
    print(result)


@pytest.mark.parametrize("model", models)
def test_predict_from_file(model):
    result = model.predict_from_file(
        file="datasets/clean_dirty/test/dirty/splash10-0a4i-8976455223-5124e0bff8404b3090d3.png",
        input="datasets/clean_dirty")
    print(result)
