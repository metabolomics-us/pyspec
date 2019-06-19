import random
from typing import List
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, Colormap
from scipy.interpolate import griddata

plt.style.use('seaborn-white')
import numpy as np
import math
import pandas as pd
import seaborn as sns
from pyspec.loader import Spectra


class Encoder:
    """
    class to easily encode spectra into a graphical form
    """

    def encode(self, spec: Spectra, width: int = 512, height: int = 512, min_mz: int = 0, max_mz: int = 2000,
               axis=False):
        # dumb approach to find max mz

        data = []

        pairs = spec.spectra.split(" ")

        # convert spectra to arrays
        for pair in pairs:
            mass, intensity = pair.split(":")

            frac, whole = math.modf(float(mass))

            data.append(
                {
                    "intensity": float(intensity),
                    "mz": float(mass),
                    "nominal": int(whole),
                    "frac": round(float(frac), 4)
                }
            )

        dataframe = pd.DataFrame(data, columns=["intensity", "mz", "nominal", "frac"])

        # group by 5 digits
        dataframe = dataframe.groupby(dataframe['mz'].apply(lambda x: round(x, 5))).sum()

        # drop data outside min and max
        dataframe = dataframe[(dataframe['nominal'] >= min_mz) & (dataframe['nominal'] <= max_mz)]

        dataframe['intensity_norm'] = dataframe['intensity'] / dataframe['intensity'].max() * 100
        dataframe['intensity_min_max'] = (dataframe['intensity'] - dataframe['intensity'].min()) / (
                dataframe['intensity'].max() - dataframe['intensity'].min())

        fix, ax = plt.subplots(2, 1)

        ax[0].scatter(dataframe['nominal'], dataframe['frac'], c=dataframe['intensity_min_max'], vmin=0, vmax=1, s=2)
        ax[0].set_xlim(min_mz, max_mz)
        ax[0].set_ylim(0, 1)

        ax[1].stem(dataframe['mz'], dataframe['intensity_min_max'])
        ax[1].set_xlim(min_mz, max_mz)
        ax[1].set_ylim(0, 1)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)

        axis = True

        if not axis:
            ax[0].axis('off')
            ax[1].axis('off')

        plt.show()
        return plt

    def encodes(self, spectra: List[Spectra], width: int = 512, height: int = 512, min_mz: int = 0,
                max_mz: int = 200, directory: str = "data/encoded"):
        """
        encodes a spectra as picture. Conceptually wise
        we will render 3 dimensions

        x is MZ as nominal
        y is MZ as accurate
        z is intensity between 0 and 100
        :param spectra:
        :return:
        """
        from splash import Spectrum, SpectrumType, Splash

        import pathlib
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        for spec in spectra:
            plt = self.encode(spec, width, height, min_mz, max_mz)
            name = Splash().splash(Spectrum(spec.spectra, SpectrumType.MS))
            plt.savefig("{}/{}.png".format(directory, name))
