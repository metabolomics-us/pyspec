from pyspec.loader import Spectra
from scipy.stats import entropy as ent

class Entropy:
    """
    computes the entropy of a spectra
    based on it's intensity values
    """

    def compute(self, spectra: Spectra) -> float:
        """
        computes the entropy of the given spectra
        :return:
        """

        intensity_values = list(map( lambda  x: float(x.split(":")[1]),spectra.spectra.split(" ")))
        return ent(intensity_values, base=2)
