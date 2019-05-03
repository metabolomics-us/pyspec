from typing import List, Optional

from pymzml.spec import Spectrum
from pyspec.similarity import msdial_similarity


class MSMSSpectrum(Spectrum):
    """
    extension of pymzml Spectrum object to support custom spectra
    """

    def __init__(self, spectrum, record_id=None, name=None, precursor_mz=None, inchikey=None):
        """
        create a new pymzml spectrum
        :param spectra:
        :param record_id:
        :param precursor_mz:
        :return:
        """

        super().__init__()

        # Handle incoming spectrum formats
        if isinstance(spectrum, str):
            # MoNA/SPLASH spectrum string format
            spectrum = [tuple(map(float, x.split(':'))) for x in spectrum.strip().split()]
        
        if isinstance(spectrum, dict):
            # Map format with m/z as key and intensity as value
            spectrum = [(k, v) for k, v in spectrum.items()]


        # Validate spectrum
        assert isinstance(spectrum, list)
        assert len(spectrum) > 0
        assert all(isinstance(x, tuple) and len(x) == 2 and
               all(isinstance(y, (int, float)) for y in x) for x in spectrum)

        # Set sorted and intensity filtered spectrum information
        # Peaks smaller than 0.5% BPI are not useful for most similarity comaprisons
        max_intensity = max(x[1] for x in spectrum)
        spectrum = [x for x in spectrum if x[1] > 0.005 * max_intensity]
        spectrum.sort(key=lambda x: x[0])

        self._peak_dict['raw'] = spectrum

        # Set optional metadata if provided
        if record_id is not None:
            self.record_id = record_id

        if name is not None:
            self.name = name

        if precursor_mz is not None:
            self._precursors = [(precursor_mz, 1)]

        if inchikey is not None:
            self.inchikey = inchikey


    @property
    def precursor(self):
        """Used for convenience, should not be used in cases where the spectrum source may be pymzml"""
        return self._precursors[0][0]


    def presence_similarity(self, library_spectrum, tolerance):
        return msdial_similarity.presence_similarity(self, library_spectrum, tolerance)

    def reverse_similarity(self, library_spectrum, tolerance, peak_count_penalty=True):
        return msdial_similarity.reverse_similarity(self, library_spectrum, tolerance, peak_count_penalty=peak_count_penalty)

    def spectral_similarity(self, library_spectrum, tolerance, peak_count_penalty=True):
        return msdial_similarity.spectral_similarity(self, library_spectrum, tolerance, peak_count_penalty=peak_count_penalty)

    def total_similarity(self, library_spectrum, ms1_tolerance, ms2_tolerance, peak_count_penalty=True):
        return msdial_similarity.total_msms_similarity(self, library_spectrum, ms1_tolerance, ms2_tolerance, peak_count_penalty=peak_count_penalty)
