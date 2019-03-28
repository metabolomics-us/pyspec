import pytest
from pymzml.spec import Spectrum

from pyspec.msms_finder import MSMSFinder

sources = [
    "http://luna.fiehnlab.ucdavis.edu/D%3A/mzml/B1_SA0001_TEDDYLipids_Pos_1RAR7_MSMS.mzml",  # should download this file
    "data/B1_SA0001_TEDDYLipids_Pos_1RAR7_MSMS.mzml"  # should load this file directly
]


@pytest.mark.parametrize("source", sources)
def test_locate(source):
    """

    :return:
    """
    print("running locate test")
    finder = MSMSFinder()

    count = 0

    def callback(msms: Spectrum, file_name: str):
        assert msms.ms_level > 1
        nonlocal count
        count = count + 1

    finder.locate(msmsSource=source, callback=callback)

    assert count > 0
