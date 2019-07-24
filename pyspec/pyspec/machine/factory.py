import importlib
from typing import Optional

from pyspec import config
from pyspec.machine.labels.generate_labels import LabelGenerator
from pyspec.machine.model.cnn import CNNClassificationModel
from pyspec.machine.spectra import Encoder


class MachineFactory:
    """
    configures everything needed to train a given model
    """

    def __init__(self, config_file: str = "machine.ini"):
        self.config_file = config_file
        self.training_config = config.config(filename=config_file, section="training")

    def load_encoder(self, name: str = None) -> Encoder:
        """
        loads the encoder for you
        :return:
        """
        encoder_config = config.config(filename=self.config_file, section="encoder")

        if name is None:
            name = encoder_config['default']

        encoder: Encoder = self.factory(name)(
            width=int(encoder_config.get("width")),
            height=int(encoder_config.get("height")),
            min_mz=float(encoder_config.get("min_mz")),
            max_mz=float(encoder_config.get("max_mz")),
            intensity_max=float(encoder_config.get("intensity_max")),
            dpi=int(encoder_config.get("dpi")),
            plot_axis=True if encoder_config.get("axis") == 'true' else False,
        )

        return encoder

    def load_generator(self) -> LabelGenerator:
        """
        loads the generator to use
        :return:
        """
        generator_config = config.config(filename=self.config_file, section="training")
        generator: LabelGenerator = self.factory(generator_config['generator'])()

        return generator

    def load_model(self, name: Optional[str] = None) -> CNNClassificationModel:
        """
        loads a model for us by name, to be utilized
        :param name:
        :return:
        """
        model_config = config.config(filename=self.config_file, section="model")
        encoder_config = config.config(filename=self.config_file, section="encoder")

        if name is None:
            name = model_config['default']

        model: CNNClassificationModel = self.factory(name)(
            width=int(model_config.get("width")),
            height=int(model_config.get("height")),
            plots=True if model_config.get("plot") == 'true' else False,
            batch_size=int(model_config['batch_size']),
            channels=3
        )

        return model

    def factory(self, module_class_string, super_cls: type = None):
        """
        :param module_class_string: full name of the class to create an object of
        :param super_cls: expected super class for validity, None if bypass
        :param kwargs: parameters to pass
        :return: class ready to be initialized
        """
        module_name, class_name = module_class_string.rsplit(".", 1)
        module = importlib.import_module(module_name)
        assert hasattr(module, class_name), "class {} is not in {}".format(class_name, module_name)
        cls = getattr(module, class_name)
        if super_cls is not None:
            assert issubclass(cls, super_cls), "class {} should inherit from {}".format(class_name, super_cls.__name__)
        return cls

    def train(self, input: str, model: Optional[CNNClassificationModel] = None, generator: Optional = None,
              gpus: int = None):
        """
        trains the model using the internal configuration
        :param model:
        :return:
        """
        train_config = config.config(filename=self.config_file, section="training")

        if model is None:
            model = self.load_model()

        if generator is None:
            generator = self.load_generator()

        model.train(
            input=input,
            generator=generator,
            test_size=float(train_config['test_size']),
            epochs=int(train_config['epoch']),
            gpus=int(train_config['gpus']) if gpus is None else int(gpus),
            verbose=int(train_config['verbose'])
        )
