from unittest import TestCase

import pytest

from pyspec.converter.mzml_to_csv import MZMLtoCSVConverter

sources = [
    (
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch1/positive/B1A_SA0001_TEDDYLipids_Pos_1RAR7_MSMS.mzml",
    "B1A_SA0001_TEDDYLipids_Pos_1RAR7_MSMS.csv"),
]


@pytest.mark.parametrize("source", sources)
def test_convert(source):
    converter = MZMLtoCSVConverter()

    converter.convert(source[0], source[1])
