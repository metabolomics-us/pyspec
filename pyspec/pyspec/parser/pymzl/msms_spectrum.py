from __future__ import annotations

from pymzml.spec import Spectrum


class MSMSSpectrum(Spectrum):
    """
    extension of pymzml Spectrum object to support custom spectra
    """

    def __init__(self, spectrum, record_id: str = None, name: str = None, precursor_mz: float = None, inchikey: str = None):
        """
        create a new pymzml spectrum
        :param spectra:
        :param record_id:
        :param precursor_mz:
        :return:
        """

        super().__init__()

        # handle incoming spectrum formats
        if isinstance(spectrum, str):
            # MoNA/SPLASH spectrum string format
            spectrum = [tuple(map(float, x.split(':'))) for x in spectrum.strip().split()]

        if isinstance(spectrum, dict):
            # map format with m/z as key and intensity as value
            spectrum = [(k, v) for k, v in spectrum.items()]

        # validate spectrum
        assert isinstance(spectrum, list)
        assert len(spectrum) > 0
        assert all(isinstance(x, tuple) and len(x) == 2 and
               all(isinstance(y, (int, float)) for y in x) for x in spectrum)

        # set sorted and intensity filtered spectrum information
        # peaks smaller than 0.5% BPI are not useful for most similarity comparisons
        max_intensity = max(x[1] for x in spectrum)
        spectrum = [x for x in spectrum if x[1] > 0.005 * max_intensity]
        spectrum.sort(key=lambda x: x[0])

        self._peak_dict['raw'] = spectrum

        # set optional metadata if provided
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
        """
        used for convenience, should not be used in cases where the spectrum source may be pymzml
        :return:
        """
        return self._precursors[0][0]


    def presence_similarity(self, library_spectrum: MSMSSpectrum, tolerance: float) -> float:
        from pyspec.similarity.msdial_similarity import presence_similarity
        return presence_similarity(self, library_spectrum, tolerance)

    def reverse_similarity(self, library_spectrum: MSMSSpectrum, tolerance: float, peak_count_penalty: bool = True) -> float:
        from pyspec.similarity.msdial_similarity import reverse_similarity
        return reverse_similarity(self, library_spectrum, tolerance, peak_count_penalty=peak_count_penalty)

    def spectral_similarity(self, library_spectrum: MSMSSpectrum, tolerance: float, peak_count_penalty: bool = True) -> float:
        from pyspec.similarity.msdial_similarity import spectral_similarity
        return spectral_similarity(self, library_spectrum, tolerance, peak_count_penalty=peak_count_penalty)

    def total_similarity(self, library_spectrum: MSMSSpectrum, ms1_tolerance: float, ms2_tolerance: float,
                         peak_count_penalty: bool = True) -> float:

        from pyspec.similarity.msdial_similarity import total_msms_similarity
        return total_msms_similarity(self, library_spectrum, ms1_tolerance, ms2_tolerance,
                                     peak_count_penalty=peak_count_penalty)

    def entropy_similarity(self, library_spectrum: MSMSSpectrum, ms2_tolerance: float) -> float:
        from pyspec.similarity.entropy_similarity import entropy_similarity_adaptor
        return entropy_similarity_adaptor(self, library_spectrum, ms2_tolerance)
