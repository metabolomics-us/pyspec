from unittest import TestCase

import pytest

from pyspec.machine.spectra import Encoder
from pyspec.machine.training.classification.binbase import BinBaseBinClassifier

classifiers = [
    BinBaseBinClassifier(name="bin_identity",
                         encoder=Encoder(),
                         output="datasets",
                         sample_count=10,
                         bin_ids=[]
                         )
]


@pytest.mark.parametrize("classifier", classifiers)
def test_classify(classifier):
    """
    executes all the classifiers
    :param classifier:
    :return:
    """
    classifier.classify()
