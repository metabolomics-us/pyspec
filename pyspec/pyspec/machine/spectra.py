from abc import abstractmethod
from typing import List, Optional, Any, Tuple

import matplotlib.pyplot as plt

plt.style.use('seaborn-white')
import math
import pandas as pd
from pyspec.loader import Spectra


class Encoder:
    """
    encodes incomming data in a graphic representation
    """

    def __init__(self, width=512, height=512, min_mz=0, max_mz=2000, plot_axis=False, intensity_max=1000, dpi=72,
                 directory: Optional = None):
        """

        :param width: width in pixel
        :param height: height in pixel
        :param min_mz: min mass
        :param max_mz: max mass
        :param plot_axis: do we want to plot axes
        :param intensity_max: max intensity
        :param dpi: resolution
        :param directory: optional directory where to store encoded data. If none just the string will be returned.
        """
        self.width = width
        self.height = height
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.axis = plot_axis
        self.intensity_max = intensity_max
        self.dpi = dpi

    def encode(self, spec: Spectra) -> Tuple[str, Any]:
        """
        encodes a spectra to a string in graphical form
        :param spec:
        :param prefix:
        :param store_meta:
        :return:
        """
        return self._encode(spec)

    def _encode(self, spec: Spectra) -> Tuple[str, List[Any]]:
        """
        encodes the given spectra
        :param spec: spectra
        :param prefix: prefix
        :param store_meta: do you also want to store the spectra string for each spectra?
        :return: encoded string of the spectra as first element, 2nd element is a list of additional computed values
        """
        # dumb approach to find max mz
        data = []

        pairs = spec.spectra.split(" ")

        # convert spectra to arrays
        for pair in pairs:
            try:
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
            except ValueError:
                pass

        try:
            dataframe = pd.DataFrame(data, columns=["intensity", "mz", "nominal", "frac"])

            # group by 5 digits
            dataframe = dataframe.groupby(dataframe['mz'].apply(lambda x: round(x, 5))).sum()

            # drop data outside min and max
            dataframe = dataframe[(dataframe['nominal'] >= self.min_mz) & (dataframe['nominal'] <= self.max_mz)]

            dataframe['intensity_min_max'] = (dataframe['intensity'] - dataframe['intensity'].min()) / (
                    dataframe['intensity'].max() - dataframe['intensity'].min())

            # formatting
            fig = plt.figure(
                figsize=(self.height / self.dpi, self.width / self.dpi), dpi=self.dpi)

            self._encode_dataframe(dataframe, fig)

            plt.tight_layout()
            fig.canvas.draw()

            spectra_string = fig.canvas.tostring_rgb()
            return spectra_string
        except ValueError:
            pass

    @abstractmethod
    def _encode_dataframe(self, dataframe, fig):
        """
        encodes the given dataframe on the figure in form of a graphic
        :param dataframe:
        :param fig:
        :return:
        """


class DualEncoder(Encoder):
    """
    this encoder encodes the data in form of 2 charts. 1 chart the actual spectra and the other a heatmap of accuracies
    """

    def _encode_dataframe(self, dataframe, fig):
        """
        encodes the given dataframe on the figure in form of a graphic
        :param dataframe:
        :param fig:
        :return:
        """
        widths = [1]
        heights = [16, 16, 1]
        specs = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, height_ratios=heights)
        ax0 = plt.subplot(specs[0, 0])
        ax1 = plt.subplot(specs[1, 0])
        ax2 = plt.subplot(specs[2, 0])
        ax0.scatter(dataframe['nominal'], dataframe['frac'], c=dataframe['intensity_min_max'], vmin=0, vmax=1, s=1)
        ax0.set_xlim(self.min_mz, self.max_mz)
        ax0.set_ylim(0, 1)
        ax1.stem(dataframe['mz'], dataframe['intensity_min_max'], markerfmt=' ', linefmt='black',
                 use_line_collection=True)
        ax1.set_xlim(self.min_mz, self.max_mz)
        ax1.set_ylim(0, 1)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.barh("intensity", dataframe['intensity'].max(), align='center', color='black')
        ax2.set_xlim(0, self.intensity_max)
        if not self.axis:
            ax0.axis('off')
            ax1.axis('off')
            ax2.axis('off')
