import pytest
from pymzml.spec import Spectrum

from pyspec.filters import MSMinLevelFilter
from pyspec.loader.binbase import BinBaseLoader
from pyspec.machine.spectra import Encoder
from pyspec.msms_finder import MSMSFinder


def test_encode():
    # 1 load some binbase spectra
    binbase = BinBaseLoader()
    data = binbase.load_spectra_for_bin_as_list(13, 1000)

    encoder = Encoder()
    encoded = encoder.encodes(data)


sources = [
    "http://luna.fiehnlab.ucdavis.edu/D%3A/mzml/B1_SA0001_TEDDYLipids_Pos_1RAR7_MSMS.mzml",  # should download this file
]


@pytest.mark.parametrize("source", sources)
def test_encode_msms(source):
    finder = MSMSFinder()

    encoder = Encoder()

    def callback(msms: Spectrum, file_name: str):
        encoded = encoder.encode(msms.convert(msms))

    finder.locate(msmsSource=source, callback=callback, filters=[MSMinLevelFilter(2)])
