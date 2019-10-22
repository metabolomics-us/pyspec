import collections
import math

from typing import Dict, List, Tuple

from pyspec.loader import Spectra
from pyspec.parser.pymzl.msms_spectrum import MSMSSpectrum


EPS_CORRECTION = 1.0e-6
MZ_ROUND_CORRECTION = 0.2


def _transform_spectrum_tuple(spectrum) -> List[Tuple[float, float]]:
    """
    transform a given spectrum from string format to a list of tuples
    :param spectrum:
    :return:
    """
    s = [tuple(map(float, x.split(':'))) for x in spectrum.split()]
    return [(mz, intensity) for mz, intensity in s if mz > 0]


def _transform_spectrum(spectrum, bin_size=None, normalize=True, scale_function=None) -> Dict[int, float]:
    """
    transform the input spectrum through binning, normalization and scaling as required
    :param spectrum:
    :param bin_size: bin width for describing sensitivity of spectrum similarity
    :param normalize: whether to normalize the binned spectrum (base peak intensity is 100)
    :param scale_function: custom function to transform the bin weights using mass and intensity
    :return:
    """

    if type(spectrum) == Spectra:
        spectrum = spectrum.spectra
    if type(spectrum) == str:
        spectrum = _transform_spectrum_tuple(spectrum)
    if type(spectrum) == MSMSSpectrum:
        spectrum = spectrum.peaks('raw')

    # build binned spectrum
    bins = collections.defaultdict(float)

    for mz, intensity in spectrum:
        if bin_size is None:
            # convert to nominal mass spectrum and include m/z correction factor
            key = int(mz + MZ_ROUND_CORRECTION)
        else:
            # generate bin index if bin size is provided
            key = int(mz / bin_size)

        bins[key] += float(intensity)

    max_intensity = max(bins.values())

    # transform spectrum through normalization and/or custom scaling
    transformed_spectrum = collections.defaultdict(float)

    for k, v in bins.items():
        if v > 0:
            transformed_spectrum[k] = v

            if normalize:
                transformed_spectrum[k] = 100 * transformed_spectrum[k] / max_intensity

            if scale_function is not None:
                # calculate mass correspoding to bin key
                if bin_size is None:
                    mass = k
                else:
                    mass = k * bin_size

                transformed_spectrum[k] = scale_function(mass, transformed_spectrum[k])

    return transformed_spectrum


def cosine_similarity(a, b, bin_size: float = None) -> float:
    """
    calculate the standard cosine similarity between two spectra
    :param a:
    :param b:
    :param bin_size: allow for a custom bin size, otherwise transform to nominal mass
    :return:
    """

    # handle different input formats
    if type(a) in [str, list, Spectra, MSMSSpectrum]:
        a = _transform_spectrum(a, bin_size=bin_size)
    if type(b) in [str, list, Spectra, MSMSSpectrum]:
        b = _transform_spectrum(b, bin_size=bin_size)

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


def composite_similarity(a, b) -> float:
    """
    calculate composite similarity between two spectra
    note: this is currently only defined for nominal mass spectra and cannot be used with custom binning
    :param a:
    :param b:
    :return:
    """

    # handle different input formats
    if type(a) in [str, list, Spectra, MSMSSpectrum]:
        a = _transform_spectrum(a)
    if type(b) in [str, list, Spectra, MSMSSpectrum]:
        b = _transform_spectrum(b)

    # identify shared ions and calculate cosine similarity
    shared_ions = sorted(x for x in set(a.keys()) & set(b.keys()) if a[x] > EPS_CORRECTION and b[x] > EPS_CORRECTION)
    cosine_sim = cosine_similarity(a, b)

    if len(shared_ions) > 1:
        ratios_a = [a[x] / a[y] for x, y in zip(shared_ions, shared_ions[1:])]
        ratios_b = [b[x] / b[y] for x, y in zip(shared_ions, shared_ions[1:])]
        combined_ratios = [x / y for x, y in zip(ratios_a, ratios_b)]

        intensity_sim = 1 + sum(x if x < 1 else 1 / x for x in combined_ratios)

        return (len(a) * cosine_sim + intensity_sim) / (len(a) + len(shared_ions))

    else:
        return cosine_sim
