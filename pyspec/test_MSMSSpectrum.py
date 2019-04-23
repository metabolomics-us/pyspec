import pytest

from pyspec.msms_spectrum import MSMSSpectrum


def test_msms_spectrum_precursor():
    spectrum = MSMSSpectrum('10:100 15:20', precursor_mz=100)
    assert(spectrum.precursor == 100)

def test_msms_spectrum_similarity():
    spectrum = MSMSSpectrum('10:100 15:20', precursor_mz=100)

    assert spectrum.presence_similarity(spectrum, 0.1) >= 0.999
    assert spectrum.reverse_similarity(spectrum, 0.1, peak_count_penalty=False) >= 0.999
    assert spectrum.similarity(spectrum, 0.1, peak_count_penalty=False) >= 0.999
    assert spectrum.total_similarity(spectrum, 0.1 peak_count_penalty=False) >= 0.999

