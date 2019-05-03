from pyspec.msms_spectrum import MSMSSpectrum


def test_msms_spectrum_precursor():
    spectrum = MSMSSpectrum('10:100 15:20', precursor_mz=100)
    assert(spectrum.precursor == 100)

def test_msms_spectrum_similarity():
    spectrum = MSMSSpectrum('10:100 15:20', precursor_mz=100)

    assert spectrum.presence_similarity(spectrum, 0.01) >= 0.999
    assert spectrum.reverse_similarity(spectrum, 0.01, peak_count_penalty=False) >= 0.999
    assert spectrum.spectral_similarity(spectrum, 0.01, peak_count_penalty=False) >= 0.999
    assert all(x >= 0.999 for x in spectrum.total_similarity(spectrum, 0.01, 0.05, peak_count_penalty=False))
