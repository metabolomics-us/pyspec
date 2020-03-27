"""
Implementations of rolling window similarity methods used in MS-DIAL for use
with accurate mass MS/MS spectra.
"""

import math
import numpy as np

from typing import List, Tuple

from pyspec.parser.pymzl.msms_spectrum import MSMSSpectrum
from pyspec.similarity.util import _get_spectrum_peak_penalty, _transform_spectrum


def _sum_ions_in_mass_window(peaks: List, starting_idx: int, focused_mz: float, tolerance: float) -> Tuple[float, int]:
    """
    find ions in peak list within the given tolerance of the given focused m/z
    :param peaks:
    :param starting_idx:
    :param focused_mz:
    :param tolerance:
    :return: the total intensity of matching ions and the updated starting index
    """

    total_intensity = 0

    for i in range(starting_idx, len(peaks)):
        if peaks[i][0] < focused_mz - tolerance:
            continue
        elif focused_mz - tolerance <= peaks[i][0] < focused_mz + tolerance:
            total_intensity += peaks[i][1]
        else:
            starting_idx = i
            break

    return total_intensity, starting_idx


def _cosine_similarity_from_peak_lists(s_mass_list: List, lib_mass_list: List,
                                       base_s: float, base_lib: float, peak_count_penalty: bool,
                                       reverse_sim: bool = False, cutoff: float = 0.01) -> float:
    """
    helper function to calculate cosine similarity for reverse and symmetric similarity methods
    :param s_mass_list: experimental ions
    :param lib_mass_list: library ions
    :param base_s: experimental base peak intensity
    :param base_lib: library base peak intensity
    :param peak_count_penalty: whether to apply a penalty for low peak counts
    :param reverse_sim: whether this is a reverse similarity calculation
    :param cutoff: relative intensity cutoff for including ions
    :return:
    """

    # normalize spectra and count abundant peaks
    s_ion_count, lib_ion_count = 0, 0
    s_sum_intensity, lib_sum_intensity = 0, 0

    for i in range(len(s_mass_list)):
        s_mass_list[i][1] /= base_s
        lib_mass_list[i][1] /= base_lib

        s_sum_intensity += s_mass_list[i][1]
        lib_sum_intensity += lib_mass_list[i][1]

        if s_mass_list[i][1] > 0.1:
            s_ion_count += 1

        if lib_mass_list[i][1] > 0.1:
            lib_ion_count += 1

    # determine penalty for low peak counts
    spectrum_penalty = _get_spectrum_peak_penalty(peak_count_penalty, lib_ion_count)

    # weights are currently unused, but can be applied later
    w_s = 1.0 / (s_sum_intensity - 0.5) if s_sum_intensity != 0.5 else 0
    w_lib = 1.0 / (lib_sum_intensity - 0.5) if lib_sum_intensity != 0.5 else 0

    # calculate similarity
    scalar_s, scalar_lib, covariance = 0, 0, 0

    for i in range(len(s_mass_list)):
        # ignore ions if intensity is less than the cutoff
        # the mass list considered depends on whether we're doing a symmetric or reverse similarity
        if reverse_sim:
            if lib_mass_list[i][1] < cutoff:
                continue
        else:
            if s_mass_list[i][1] < cutoff:
                continue

        scalar_s += s_mass_list[i][0] * s_mass_list[i][1]
        scalar_lib += lib_mass_list[i][0] * lib_mass_list[i][1]
        covariance += math.sqrt(s_mass_list[i][1] * lib_mass_list[i][1]) * s_mass_list[i][0]

    if scalar_s == 0 or scalar_lib == 0:
        return 0
    else:
        return pow(covariance, 2) / scalar_s / scalar_lib * spectrum_penalty



def gaussian_similarity(actual, reference, tolerance: float):
    """
    generates a Gaussian similarity between actual and reference values (or numpy arrays) with the given
    tolerance as the allowable spread
    :param actual:
    :param reference:
    :param tolerance:
    :return:
    """

    return np.exp(-0.5 * pow((actual - reference) / tolerance, 2))


def presence_similarity(s, lib, tolerance: float):
    """
    calculate the Jaccard/presence similarity of ions in two spectra
    :param s: experimental spectrum
    :param lib: library spectrum
    :param tolerance: tolerance ion comparisons
    :return:
    """

    s_peaks = _transform_spectrum(s)
    lib_peaks = _transform_spectrum(lib)

    min_mz = lib_peaks[0][0]
    max_mz = lib_peaks[-1][0]
    focused_mz = min_mz
    
    max_lib_intensity = max(intensity for mz, intensity in lib_peaks)
    
    i_s, i_lib = 0, 0
    s_counter, lib_counter = 0, 0
    
    while focused_mz <= max_mz:
        # find library ions within tolerance of the focused m/z
        sum_lib, i_lib = _sum_ions_in_mass_window(lib_peaks, i_lib, focused_mz, tolerance)

        if sum_lib >= 0.01 * max_lib_intensity:
            lib_counter += 1
                
        # find unknown ions within tolerance of the focused m/z
        sum_s, i_s = _sum_ions_in_mass_window(s_peaks, i_s, focused_mz, tolerance)
        
        if sum_s > 0 and sum_lib >= 0.01 * max_lib_intensity:
            s_counter += 1
        
        # go to the next focused mass
        if focused_mz + tolerance > max_mz:
            break
        else:
            focused_mz = lib_peaks[i_lib][0]

    # calculate Jaccard similarity
    if lib_counter == 0:
        return 0
    else:
        return s_counter / lib_counter


def reverse_similarity(s, lib, tolerance: float, peak_count_penalty: bool = True):
    """
    calculate the reverse similarity between two spectra (i.e., the cosine similarity restricted to ions
    present in the lib spectrum). note that this is not a symmetric similarity method
    :param s: experimental spectrum
    :param lib: library spectrum
    :param tolerance: tolerance ion comparisons
    :param peak_count_penalty: whether to apply a penalty for low peak counts
    :return:
    """

    s_peaks = _transform_spectrum(s)
    lib_peaks = _transform_spectrum(lib)
    
    min_mz = lib_peaks[0][0]
    max_mz = lib_peaks[-1][0]
    focused_mz = min_mz
    
    i_s, i_lib = 0, 0
    base_s, base_lib = 0, 0
    counter = 0
    
    s_mass_list, lib_mass_list = [], []
    
    while focused_mz <= max_mz:
        # find library ions within tolerance of the focused m/z
        sum_lib, i_lib = _sum_ions_in_mass_window(lib_peaks, i_lib, focused_mz, tolerance)
                
        # find unknown ions within tolerance of the focused m/z
        sum_s, i_s = _sum_ions_in_mass_window(s_peaks, i_s, focused_mz, tolerance)
        
        # add focused mass intensities
        s_mass_list.append([focused_mz, sum_s])
        if sum_s > base_s:
            base_s = sum_s
        
        lib_mass_list.append([focused_mz, sum_lib])
        if sum_lib > base_lib:
            base_lib = sum_lib
        
        if sum_s > 0:
            counter += 1
        
        # go to the next focused mass
        if focused_mz + tolerance > max_mz:
            break
        else:
            focused_mz = lib_peaks[i_lib][0]

    if base_s == 0 or base_lib == 0:
        # return a score of 0 if no matching ions are found
        return 0
    else:
        # calculate similarity
        return _cosine_similarity_from_peak_lists(s_mass_list, lib_mass_list, base_s, base_lib,
                                                  peak_count_penalty, reverse_sim=True)


def spectral_similarity(s, lib, tolerance: float, peak_count_penalty: bool = True):
    """
    calculate the cosine similarity between two spectra
    :param s: experimental spectrum
    :param lib: library spectrum
    :param tolerance: tolerance ion comparisons
    :param peak_count_penalty: whether to apply a penalty for low peak counts
    :return:
    """

    s_peaks = _transform_spectrum(s)
    lib_peaks = _transform_spectrum(lib)
    
    min_mz = min(s_peaks[0][0], lib_peaks[0][0])
    max_mz = max(s_peaks[-1][0], lib_peaks[-1][0])
    focused_mz = min_mz
    
    i_s, i_lib = 0, 0
    base_s, base_lib = 0, 0
    
    s_mass_list, lib_mass_list = [], []
    
    while focused_mz <= max_mz:
        # find library ions within tolerance of the focused m/z
        sum_lib, i_lib = _sum_ions_in_mass_window(lib_peaks, i_lib, focused_mz, tolerance)

        # find unknown ions within tolerance of the focused m/z
        sum_s, i_s = _sum_ions_in_mass_window(s_peaks, i_s, focused_mz, tolerance)
        
        # add focused mass intensities
        s_mass_list.append([focused_mz, sum_s])
        if sum_s > base_s:
            base_s = sum_s
        
        lib_mass_list.append([focused_mz, sum_lib])
        if sum_lib > base_lib:
            base_lib = sum_lib
        
        # go to the next focused mass
        if focused_mz + tolerance > max_mz:
            break
        
        if focused_mz + tolerance > lib_peaks[i_lib][0] and focused_mz + tolerance <= s_peaks[i_s][0]:
            focused_mz = s_peaks[i_s][0]
        elif focused_mz + tolerance <= lib_peaks[i_lib][0] and focused_mz + tolerance > s_peaks[i_s][0]:
            focused_mz = lib_peaks[i_lib][0]
        else:
            focused_mz = min(s_peaks[i_s][0], lib_peaks[i_lib][0])

    if base_s == 0 or base_lib == 0:
        # return a score of 0 if no matching ions are found
        return 0
    else:
        # calculate similarity
        return _cosine_similarity_from_peak_lists(s_mass_list, lib_mass_list, base_s, base_lib,
                                                  peak_count_penalty)


def total_msms_similarity(s, lib, ms1_tolerance, ms2_tolerance, s_precursor: float = None,
                          lib_precursor: float = None, peak_count_penalty: bool = True):

    """
    calculate the total similarity by combining multiple similarity metrics with custom scaling
    :param s: experimental spectrum
    :param lib: library spectrum
    :param ms1_tolerance: tolerance for precursor m/z
    :param ms2_tolerance: tolerance for MS/MS ions
    :param s_precursor: precursor for experimental spectrum (only required when spectrum is not a MSMSSpectrum)
    :param lib_precursor: precursor for library spectrum (only required when spectrum is not a MSMSSpectrum)
    :param peak_count_penalty: whether to apply a penalty for low peak counts
    :return:
    """

    if type(s) == MSMSSpectrum and type(lib) == MSMSSpectrum:
        s_precursor = s.precursors[0][0]
        lib_precursor = lib.precursors[0][0]
    elif s_precursor is None or lib_precursor is None:
        raise Exception('no precursor provided implicitly through MSMSSpectrum objects or explicitly in function call')

    # scaling factors
    dot_product_factor = 3
    reverse_dot_product_factor = 2
    presence_percentage_factor = 1

    msms_factor = 2
    mass_factor = 1

    # raw similarities
    accurate_mass_sim = gaussian_similarity(s_precursor, lib_precursor, ms1_tolerance)

    spectral_sim = spectral_similarity(s, lib, ms2_tolerance, peak_count_penalty=peak_count_penalty)
    reverse_sim = reverse_similarity(s, lib, ms2_tolerance, peak_count_penalty=peak_count_penalty)
    presence_sim = presence_similarity(s, lib, ms2_tolerance)

    # calculate MS/MS similarity
    msms_sim = (dot_product_factor * spectral_sim + reverse_dot_product_factor * reverse_sim + presence_percentage_factor * presence_sim)
    msms_sim /= (dot_product_factor + reverse_dot_product_factor + presence_percentage_factor)

    # calculate total similarity
    total_similarity = (msms_factor * msms_sim + mass_factor * accurate_mass_sim) / (msms_factor + mass_factor)

    # return all similarity values
    return spectral_sim, reverse_sim, presence_sim, msms_sim, accurate_mass_sim, total_similarity
