from unittest import TestCase

from pyspec.loader.binbase import BinBaseLoader


def test_load_spectra_for_bin():
    binbase = BinBaseLoader()
    data = binbase.load_spectra_for_bin_as_list(1, 100)

    assert len(data) == 100
