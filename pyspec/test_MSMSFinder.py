import pytest
from pymzml.spec import Spectrum

from pyspec.filters import MSMinLevelFilter
from pyspec.msms_finder import MSMSFinder

sources = [
    "http://luna.fiehnlab.ucdavis.edu/D%3A/mzml/B1_SA0001_TEDDYLipids_Pos_1RAR7_MSMS.mzml",  # should download this file
    "data/B1_SA0001_TEDDYLipids_Pos_1RAR7_MSMS.mzml"  # should load this file directly
]


@pytest.mark.parametrize("source", sources)
def test_locate_without_filter(source):
    """

    :return:
    """
    finder = MSMSFinder()

    count = 0

    def callback(msms: Spectrum, file_name: str):
        nonlocal count
        count = count + 1

    finder.locate(msmsSource=source, callback=callback)

    assert count > 0


@pytest.mark.parametrize("source", sources)
def test_locate_with_msms_filter(source):
    """

    :return:
    """
    finder = MSMSFinder()

    count = 0

    def callback(msms: Spectrum, file_name: str):
        nonlocal count
        count = count + 1
        assert msms.ms_level > 1

    finder.locate(msmsSource=source, callback=callback, filters=[MSMinLevelFilter(2)])
    assert count > 0

@pytest.mark.parametrize("source", sources)
def test_convert(source):
    """

    :return:
    """
    finder = MSMSFinder()

    count = 0

    def callback(msms: Spectrum, file_name: str):
        nonlocal count
        count = count + 1

        converted = msms.convert(msms)


    finder.locate(msmsSource=source, callback=callback, filters=[MSMinLevelFilter(2)])
    assert count > 0

@pytest.mark.parametrize("source", sources)
def test_locate_with_msms_and_compute_count(source):
    """

    :return:
    """
    finder = MSMSFinder()

    count = {}

    def callback(msms: Spectrum, file_name: str):
        nonlocal count

        if msms.ms_level not in count:
            count[msms.ms_level] = 0

        count[msms.ms_level] = count[msms.ms_level] + 1

    finder.locate(msmsSource=source, callback=callback)

    for key in count:
        print(f"the obsderved count of spectra with level {key} in file {source} was {count[key]}")
