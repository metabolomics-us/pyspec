from pyspec.parser.pymzl.msms_spectrum import MSMSSpectrum
from pyspec.similarity.entropy_similarity import entropy_similarity
import numpy as np


def test_example():
    spec_query = np.array([[69.071, 7.917962], [86.066, 1.021589], [86.0969, 100.0]])
    spec_reference = np.array([[41.04, 37.16], [69.07, 66.83], [86.1, 999.0]])
    similarity_score = entropy_similarity(spec_query, spec_reference, ms2_tolerance=0.05, precursor_cutoff=100)
    assert similarity_score > 0.9


def test_msms_spectrum_similarity():
    a = MSMSSpectrum('10:100', precursor_mz=100)
    b = MSMSSpectrum('10.5:100', precursor_mz=100)

    assert a.entropy_similarity(b, 0.51) >= 0.999
    assert a.entropy_similarity(b, 0.49) == 0


def test_msms_spectrum_similarity_2():
    a = MSMSSpectrum('10:100 20:50', precursor_mz=100)
    b = MSMSSpectrum('10:100', precursor_mz=100)

    assert a.entropy_similarity(a, 1) >= 0.999
    assert b.entropy_similarity(b, 1) >= 0.999
    assert b.entropy_similarity(a, 1) < 1
