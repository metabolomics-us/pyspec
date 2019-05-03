"""
Implementations of rolling window similarity methods used in MS-DIAL for use
with accurate mass MS/MS spectra.

Requires all inputs match the format 
"""

import math


def gaussian_similarity(actual, reference, tolerance):
    return math.exp(-0.5 * pow((actual - reference) / tolerance, 2))


def presence_similarity(s, lib, tolerance):
    s_peaks = s.peaks('raw')
    lib_peaks = lib.peaks('raw')

    min_mz = lib_peaks[0][0]
    max_mz = lib_peaks[-1][0]
    focused_mz = min_mz
    
    max_lib_intensity = max(intensity for mz, intensity in lib_peaks)
    
    i_s, i_lib = 0, 0
    s_counter, lib_counter = 0, 0
    
    while focused_mz <= max_mz:
        # Find library ions within tolerance of the focused m/z
        sum_lib = 0
        
        for i in range(i_lib, len(lib_peaks)):
            if lib_peaks[i][0] < focused_mz - tolerance:
                continue
            elif focused_mz - tolerance <= lib_peaks[i][0] < focused_mz + tolerance:
                sum_lib += lib_peaks[i][1]
            else:
                i_lib = i
                break
        
        if sum_lib >= 0.01 * max_lib_intensity:
            lib_counter += 1
                
        # Find unknown ions within tolerance of the focused m/z
        sum_s = 0
        
        for i in range(i_s, len(s_peaks)):
            if s_peaks[i][0] < focused_mz - tolerance:
                continue
            elif focused_mz - tolerance <= s_peaks[i][0] < focused_mz + tolerance:
                sum_s += s_peaks[i][1]
            else:
                i_s = i
                break
        
        if sum_s > 0 and sum_lib >= 0.01 * max_lib_intensity:
            s_counter += 1
        
        # Go to the next focused mass
        if focused_mz + tolerance > max_mz:
            break
        else:
            focused_mz = lib_peaks[i_lib][0]
    
    if lib_counter == 0:
        return 0
    else:
        return s_counter / lib_counter


def reverse_similarity(s, lib, tolerance, peak_count_penalty=True):
    s_peaks = s.peaks('raw')
    lib_peaks = lib.peaks('raw')
    
    min_mz = lib_peaks[0][0]
    max_mz = lib_peaks[-1][0]
    focused_mz = min_mz
    
    i_s, i_lib = 0, 0
    base_s, base_lib = 0, 0
    counter = 0
    
    s_mass_list, lib_mass_list = [], []
    
    while focused_mz <= max_mz:
        # Find library ions within tolerance of the focused m/z
        sum_lib = 0
        
        for i in range(i_lib, len(lib_peaks)):
            if lib_peaks[i][0] < focused_mz - tolerance:
                continue
            elif focused_mz - tolerance <= lib_peaks[i][0] < focused_mz + tolerance:
                sum_lib += lib_peaks[i][1]
            else:
                i_lib = i
                break
                
        # Find unknown ions within tolerance of the focused m/z
        sum_s = 0
        
        for i in range(i_s, len(s_peaks)):
            if s_peaks[i][0] < focused_mz - tolerance:
                continue
            elif focused_mz - tolerance <= s_peaks[i][0] < focused_mz + tolerance:
                sum_s += s_peaks[i][1]
            else:
                i_s = i
                break
        
        # Add focused mass intensities
        s_mass_list.append([focused_mz, sum_s])
        if sum_s > base_s:
            base_s = sum_s
        
        lib_mass_list.append([focused_mz, sum_lib])
        if sum_lib > base_lib:
            base_lib = sum_lib
        
        if sum_s > 0:
            counter += 1
        
        # Go to the next focused mass
        if focused_mz + tolerance > max_mz:
            break
        else:
            focused_mz = lib_peaks[i_lib][0]
        
    # Return a score of 0 if no matching ions are found
    if base_s == 0 or base_lib == 0:
        return 0
    
    # Normalize spectra and count abundant peaks
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
    
    # Determine penalty for low peak counts
    if not peak_count_penalty:
        spectrum_penalty = 1
    elif lib_ion_count == 1:
        spectrum_penalty = 0.75
    elif lib_ion_count == 2:
        spectrum_penalty = 0.88
    elif lib_ion_count == 3:
        spectrum_penalty = 0.94
    elif lib_ion_count == 4:
        spectrum_penalty = 0.97
    else:
        spectrum_penalty = 1

    # Calculate similarity
    w_s = 1.0 / (s_sum_intensity - 0.5) if s_sum_intensity != 0.5 else 0
    w_lib = 1.0 / (lib_sum_intensity - 0.5) if lib_sum_intensity != 0.5 else 0
    
    scalar_s, scalar_lib, covariance = 0, 0, 0
    cutoff = 0.01
    
    for i in range(len(s_mass_list)):
        if s_mass_list[i][1] < cutoff:
            continue
        
        scalar_s += s_mass_list[i][0] * s_mass_list[i][1]
        scalar_lib += lib_mass_list[i][0] * lib_mass_list[i][1]
        covariance += math.sqrt(s_mass_list[i][1] * lib_mass_list[i][1]) * s_mass_list[i][0]
    
    if scalar_s == 0 or scalar_lib == 0:
        return 0
    else:
        return pow(covariance, 2) / scalar_s / scalar_lib * spectrum_penalty


def spectral_similarity(s, lib, tolerance, peak_count_penalty=True):
    s_peaks = s.peaks('raw')
    lib_peaks = lib.peaks('raw')
    
    min_mz = min(s_peaks[0][0], lib_peaks[0][0])
    max_mz = max(s_peaks[-1][0], lib_peaks[-1][0])
    focused_mz = min_mz
    
    i_s, i_lib = 0, 0
    base_s, base_lib = 0, 0
    
    s_mass_list, lib_mass_list = [], []
    
    while focused_mz <= max_mz:
        # Find unknown ions within tolerance of the focused m/z
        sum_s = 0
        
        for i in range(i_s, len(s_peaks)):
            if s_peaks[i][0] < focused_mz - tolerance:
                continue
            elif focused_mz - tolerance <= s_peaks[i][0] < focused_mz + tolerance:
                sum_s += s_peaks[i][1]
            else:
                i_s = i
                break
        
        # Find library ions within tolerance of the focused m/z
        sum_lib = 0
        
        for i in range(i_lib, len(lib_peaks)):
            if lib_peaks[i][0] < focused_mz - tolerance:
                continue
            elif focused_mz - tolerance <= lib_peaks[i][0] < focused_mz + tolerance:
                sum_lib += lib_peaks[i][1]
            else:
                i_lib = i
                break
        
        # Add focused mass intensities
        s_mass_list.append([focused_mz, sum_s])
        if sum_s > base_s:
            base_s = sum_s
        
        lib_mass_list.append([focused_mz, sum_lib])
        if sum_lib > base_lib:
            base_lib = sum_lib
        
        # Go to the next focused mass
        if focused_mz + tolerance > max_mz:
            break
        
        if focused_mz + tolerance > lib_peaks[i_lib][0] and focused_mz + tolerance <= s_peaks[i_s][0]:
            focused_mz = s_peaks[i_s][0]
        elif focused_mz + tolerance <= lib_peaks[i_lib][0] and focused_mz + tolerance > s_peaks[i_s][0]:
            focused_mz = lib_peaks[i_lib][0]
        else:
            focused_mz = min(s_peaks[i_s][0], lib_peaks[i_lib][0])
        
    # Return a score of 0 if no matching ions are found
    if base_s == 0 or base_lib == 0:
        return 0
    
    # Normalize spectra and count abundant peaks
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
    
    # Determine penalty for low peak counts
    if not peak_count_penalty:
        spectrum_penalty = 1
    elif lib_ion_count == 1:
        spectrum_penalty = 0.75
    elif lib_ion_count == 2:
        spectrum_penalty = 0.88
    elif lib_ion_count == 3:
        spectrum_penalty = 0.94
    elif lib_ion_count == 4:
        spectrum_penalty = 0.97
    else:
        spectrum_penalty = 1
    
    # Calculate similarity
    w_s = 1.0 / (s_sum_intensity - 0.5) if s_sum_intensity != 0.5 else 0
    w_lib = 1.0 / (lib_sum_intensity - 0.5) if lib_sum_intensity != 0.5 else 0
    
    scalar_s, scalar_lib, covariance = 0, 0, 0
    cutoff = 0.01
    
    for i in range(len(s_mass_list)):
        if s_mass_list[i][1] < cutoff:
            continue
        
        scalar_s += s_mass_list[i][0] * s_mass_list[i][1]
        scalar_lib += lib_mass_list[i][0] * lib_mass_list[i][1]
        covariance += math.sqrt(s_mass_list[i][1] * lib_mass_list[i][1]) * s_mass_list[i][0]
    
    if scalar_s == 0 or scalar_lib == 0:
        return 0
    else:
        return pow(covariance, 2) / scalar_s / scalar_lib * spectrum_penalty


def total_msms_similarity(s, lib, ms1_tolerance, ms2_tolerance, peak_count_penalty=True):
    # Scaling factors
    dot_product_factor = 3
    reverse_dot_product_factor = 2
    presence_percentage_factor = 1

    msms_factor = 2
    rt_factor = 1
    mass_factor = 1

    # Raw similarities
    accurate_mass_sim = gaussian_similarity(s.precursors[0][0], lib.precursors[0][0], ms1_tolerance)

    spectral_sim = spectral_similarity(s, lib, ms2_tolerance, peak_count_penalty=peak_count_penalty)
    reverse_sim = reverse_similarity(s, lib, ms2_tolerance, peak_count_penalty=peak_count_penalty)
    presence_sim = presence_similarity(s, lib, ms2_tolerance)

    msms_sim = (dot_product_factor * spectral_sim + reverse_dot_product_factor * reverse_sim + presence_percentage_factor * presence_sim)
    msms_sim /= (dot_product_factor + reverse_dot_product_factor + presence_percentage_factor)

    total_similarity = (msms_factor * msms_sim + mass_factor * accurate_mass_sim) / (msms_factor + mass_factor)
    return spectral_sim, reverse_sim, presence_sim, msms_sim, accurate_mass_sim, total_similarity
