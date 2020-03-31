import collections

from typing import Callable, Dict, List, Tuple

from pyspec.loader import Spectra
from pyspec.parser.pymzl.msms_spectrum import MSMSSpectrum


EPS_CORRECTION = 1.0e-6
MZ_ROUND_CORRECTION = 0.2


def _transform_spectrum_tuple(spectrum) -> List[Tuple[float, float]]:
    """
    transform a given spectrum from string format to a list of tuples and remove ions with zero intensity
    :param spectrum:
    :return:
    """
    s = [tuple(map(float, x.split(':'))) for x in spectrum.split()]
    return [(mz, intensity) for mz, intensity in s if mz > 0]

def _transform_spectrum(spectrum) -> List:
    """
    transform a given spectrum from given format to a list of tuples/lists
    :param spectrum:
    :return:
    """

    if type(spectrum) == Spectra:
        # handle pyspec spectrum format
        spectrum = spectrum.spectra

    if type(spectrum) == str:
        # handle spectrum string format
        return _transform_spectrum_tuple(spectrum)
    elif type(spectrum) == MSMSSpectrum:
        # handle pymzml spectrum format
        return spectrum.peaks('raw')
    elif type(spectrum) == list and all(type(x) == list for x in spectrum):
        # sort list format
        return sorted(spectrum)
    elif type(spectrum) in [dict, collections.defaultdict]:
        # handle dict format
        return sorted([(k, v) for k, v in spectrum.items()])
    else:
        raise Exception(f'invalid spectrum type: {type(spectrum)}')

def _transform_binned_spectrum(spectrum, bin_size: float = None, normalize: bool = True,
                               scale_function: Callable = None) -> Dict[int, float]:
    """
    transform the input spectrum through binning, normalization and scaling as required
    :param spectrum:
    :param bin_size: bin width for describing sensitivity of spectrum similarity
    :param normalize: whether to normalize the binned spectrum (base peak intensity is 100)
    :param scale_function: custom function to transform the bin weights using mass and intensity
    :return:
    """

    # transform input spectrum
    spectrum = _transform_spectrum(spectrum)

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
                # calculate mass corresponding to bin key
                if bin_size is None:
                    mass = k
                else:
                    mass = k * bin_size

                transformed_spectrum[k] = scale_function(mass, transformed_spectrum[k])

    return transformed_spectrum


def _get_spectrum_peak_penalty(peak_count_penalty: bool, ion_count: int) -> float:
    """
    get the similarity penalty based on the provided ion count
    :param peak_count_penalty:
    :param ion_count:
    :return:
    """

    if not peak_count_penalty:
        return 1
    elif ion_count == 1:
        return 0.75
    elif ion_count == 2:
        return 0.88
    elif ion_count == 3:
        return 0.94
    elif ion_count == 4:
        return 0.97
    else:
        return 1

