from pyspec.similarity.nominal_similarity import *

def test_identical_spectra_with_one_ion():
    s = '169:3169812'
    assert cosine_similarity(s, s) >= 0.999
    assert composite_similarity(s, s) >= 0.999

def test_identical_spectra_with_many_ion():
    s = '169:3169812 155:44809 199:282429'
    assert cosine_similarity(s, s) >= 0.999
    assert composite_similarity(s, s) >= 0.999

def test_overlapping_spectra():
    s1 = '10:100 15:20'
    s2 = '10:100 15:20 20:5'
    assert cosine_similarity(s1, s2) >= 0.99
    assert composite_similarity(s1, s2) > 0.94

def test_orthogonal_spectra():
    s1 = '10:100 15:20'
    s2 = '12:100 17:25 20:5'
    assert cosine_similarity(s1, s2) < 0.001
    assert composite_similarity(s1, s2) < 0.001
