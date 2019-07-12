import os
from abc import abstractmethod

from pyspec.machine.spectra import Encoder


class Classifier:
    """
    this is a helper class to provide a simple way to generate training data
    in the exspected directory format from different external classes.
    """

    def __init__(self, name: str, encoder: Encoder, output: str):
        """"""
        assert name is not None, "please specify a name"
        assert encoder is not None, "please specify an encoder"
        assert os.path.exists(output), "please ensure that the output path exists. Path is {}".format(output)
        assert os.path.isdir(output), "please ensure that the output path is a directory. Path is {}".format(output)

        self.name = name
        self.output = output
        self.encoder = encoder
        self.encoder.directory = "{}/{}/train".format(output, name)

    @abstractmethod
    def classify(self, include_test_data: bool = False):
        """
        does the actual classification for us and if requested also generates test data
        :return:
        """
