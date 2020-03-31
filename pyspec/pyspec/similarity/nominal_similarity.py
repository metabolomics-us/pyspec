import math

from typing import Callable

from pyspec.similarity.util import SpecTypes, EPS_CORRECTION, _transform_binned_spectrum


def cosine_similarity(a: SpecTypes, b: SpecTypes, bin_size: float = None, scale_function: Callable = None) -> float:
    """
    calculate the standard cosine similarity between two spectra
    :param a:
    :param b:
    :param bin_size: allow for a custom bin size, otherwise transform to nominal mass
    :param scale_function: custom function to transform the bin weights using mass and intensity
    :return:
    """

    # handle different input formats
    if type(a) != dict:
        a = _transform_binned_spectrum(a, bin_size=bin_size, scale_function=scale_function)
    if type(b) != dict:
        b = _transform_binned_spectrum(b, bin_size=bin_size, scale_function=scale_function)

    # calculate norm for each spectrum using all ions
    normA = sum(v * v for v in a.values())
    normB = sum(v * v for v in b.values())

    # calculate cosine similarity
    if normA == 0 or normB == 0:
        return 0
    else:
        shared_ions = sorted(set(a.keys()) & set(b.keys()))
        product = math.pow(sum(a[k] * b[k] for k in shared_ions), 2)

        return product / normA / normB


def composite_similarity(a: SpecTypes, b: SpecTypes, bin_size: float = None, scale_function: Callable = None) -> float:
    """
    calculate composite similarity between two spectra
    note: this is currently only defined for nominal mass spectra and cannot be used with custom binning
    :param a:
    :param b:
    :param bin_size: allow for a custom bin size, otherwise transform to nominal mass
    :param scale_function: custom function to transform the bin weights using mass and intensity
    :return:
    """

    # handle different input formats
    if type(a) != dict:
        a = _transform_binned_spectrum(a, bin_size=bin_size, scale_function=scale_function)
    if type(b) != dict:
        b = _transform_binned_spectrum(b, bin_size=bin_size, scale_function=scale_function)

    # identify shared ions and calculate cosine similarity
    shared_ions = sorted(x for x in set(a.keys()) & set(b.keys()) if a[x] > EPS_CORRECTION and b[x] > EPS_CORRECTION)
    cosine_sim = cosine_similarity(a, b)

    if len(shared_ions) > 1:
        # calculate shared ion ratios
        ratios_a = [a[x] / a[y] for x, y in zip(shared_ions, shared_ions[1:])]
        ratios_b = [b[x] / b[y] for x, y in zip(shared_ions, shared_ions[1:])]
        combined_ratios = [x / y for x, y in zip(ratios_a, ratios_b)]

        intensity_sim = 1 + sum(x if x < 1 else 1 / x for x in combined_ratios)

        return (len(a) * cosine_sim + intensity_sim) / (len(a) + len(shared_ions))

    else:
        return cosine_sim
