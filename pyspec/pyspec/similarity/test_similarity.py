from pyspec.parser.pymzl.msms_spectrum import MSMSSpectrum


def test_msms_spectrum_similarity():
    a = MSMSSpectrum('10:100', precursor_mz=100)
    b = MSMSSpectrum('10.5:100', precursor_mz=100)

    assert a.presence_similarity(b, 0.51) >= 0.999
    assert a.presence_similarity(b, 0.49) == 0
    assert a.reverse_similarity(b, 0.51, peak_count_penalty=False) >= 0.999
    assert a.reverse_similarity(b, 0.49, peak_count_penalty=False) == 0
    assert a.spectral_similarity(b, 0.51, peak_count_penalty=False) >= 0.999
    assert a.spectral_similarity(b, 0.49, peak_count_penalty=False) == 0


def test_peak_count_penalty():
    a = MSMSSpectrum('10:100', precursor_mz=100)
    assert a.spectral_similarity(a, 1, peak_count_penalty=False) > a.spectral_similarity(a, 1)


def test_presence_similarity():
    a = MSMSSpectrum('10:100 20:50', precursor_mz=100)
    b = MSMSSpectrum('10:100', precursor_mz=100)

    assert a.presence_similarity(a, 1) >= 0.999
    assert b.presence_similarity(b, 1) >= 0.999
    assert b.presence_similarity(a, 1) < 1


def test_reverse_similarity():
    a = MSMSSpectrum('10:100 20:50', precursor_mz=100)
    b = MSMSSpectrum('10:100', precursor_mz=100)

    assert a.reverse_similarity(b, 1, peak_count_penalty=False) >= b.reverse_similarity(a, 1, peak_count_penalty=False)
