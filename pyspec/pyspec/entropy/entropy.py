from typing import Tuple

from sklearn.preprocessing import MinMaxScaler

from pyspec.loader import Spectra
from scipy.stats import entropy as ent


class Entropy:
    """
    computes the entropy of a spectra
    based on it's intensity values
    """

    def compute(self, spectra: Spectra, min_intensity: float = 0) -> Tuple[float, int]:
        """
        computes the entropy of the given spectra
        :param: spectra the given spectra
        param: min_intensity minimum required intensity
        :return:
        """

        intensity_values = list(map(lambda x: float(x.split(":")[1]), spectra.spectra.split(" ")))
        scaled_values = []
        max_value = max(intensity_values)

        for x in intensity_values:
            scaled_values.append(x / max_value * 100)

        intensity_values = list(filter(lambda x: x > min_intensity, scaled_values))
        return ent(intensity_values, base=2), len(intensity_values)
