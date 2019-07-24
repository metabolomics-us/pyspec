from pyspec.machine.factory import MachineFactory
from pyspec.machine.labels.generate_labels import LabelGenerator
from pyspec.machine.model.application import Resnet50CNNModel
from pyspec.machine.model.cnn import CNNClassificationModel


def test_load_encoder():
    machine = MachineFactory()
    encoder = machine.load_encoder()
    assert encoder.width == 500
    assert encoder.height == 500
    assert encoder.axis is False
    assert encoder.intensity_max == 1000
    assert encoder.min_mz == 0
    assert encoder.max_mz == 2000
    assert encoder.dpi == 72


def test_load_generator():
    machine = MachineFactory()
    generator = machine.load_generator()

    assert isinstance(generator, LabelGenerator)


def test_load_model():
    machine = MachineFactory()

    model = machine.load_model("pyspec.machine.model.application.Resnet50CNNModel")

    assert isinstance(model, CNNClassificationModel)
    assert isinstance(model, Resnet50CNNModel)


def test_load_default_model():
    machine = MachineFactory()

    model = machine.load_model()

    assert isinstance(model, CNNClassificationModel)
    assert isinstance(model, Resnet50CNNModel)


def test_train_model():
    machine = MachineFactory()
    machine.train("datasets/clean_dirty")
