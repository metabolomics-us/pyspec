from typing import List

import matplotlib.pyplot as plt

plt.style.use('seaborn-white')
import math
import pandas as pd
from pyspec.loader import Spectra


class Encoder:
    """
    class to easily encode spectra into a graphical form, to be used for machine learning
    """

    def encode(self, spec: Spectra, width: int = 512, height: int = 512, min_mz: int = 0, max_mz: int = 2000,
               axis=False, intensity_max=1000):
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

        dataframe['intensity_min_max'] = (dataframe['intensity'] - dataframe['intensity'].min()) / (
                dataframe['intensity'].max() - dataframe['intensity'].min())

        # formatting
        fig = plt.figure(constrained_layout=True)

        widths = [1]
        heights = [16, 16, 1]
        specs = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights)

        ax0 = plt.subplot(specs[0, 0])
        ax1 = plt.subplot(specs[1, 0])
        ax2 = plt.subplot(specs[2, 0])

        ax0.scatter(dataframe['nominal'], dataframe['frac'], c=dataframe['intensity_min_max'], vmin=0, vmax=1, s=2)
        ax0.set_xlim(min_mz, max_mz)
        ax0.set_ylim(0, 1)

        ax1.stem(dataframe['mz'], dataframe['intensity_min_max'], markerfmt=' ', linefmt='black')
        ax1.set_xlim(min_mz, max_mz)
        ax1.set_ylim(0, 1)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        ax2.barh("intensity", dataframe['intensity'].max(), align='center', color='black')
        ax2.set_xlim(0, intensity_max)

        if not axis:
            ax0.axis('off')
            ax1.axis('off')
            ax2.axis('off')

        plt.tight_layout()
#        plt.show()
        return plt

    def encodes(self, spectra: List[Spectra], width: int = 512, height: int = 512, min_mz: int = 0,
                max_mz: int = 200, directory: str = "data/encoded", axis=None, max_intensity=1000):
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
            plt = self.encode(spec, width, height, min_mz, max_mz, axis, max_intensity)
            name = Splash().splash(Spectrum(spec.spectra, SpectrumType.MS))
            plt.savefig("{}/{}.png".format(directory, name),dpi=self.dpi)
