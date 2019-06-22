from abc import abstractmethod

from pymzml.spec import Spectrum


class Filter:
    """
    a simple filter if we want to inspect a given spectra to save processing overhead
    you should provide subclasses of it, to specify it's except behavior
    """

    @abstractmethod
    def accept(self, spectra: Spectrum) -> bool:
        """
        return true if filter condition is met
        :param spectra:
        :return:
        """


class MSMinLevelFilter(Filter):
    """
    filters out spectra, which level is below the specified minimum level
    """

    def accept(self, spectra: Spectrum) -> bool:
        return spectra.ms_level >= self._min_ms_level

    def __init__(self, min_ms_level: int = 2):
        self._min_ms_level = min_ms_level
