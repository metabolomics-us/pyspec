import matplotlib.pyplot as plt
from typing import List, Optional

plt.style.use('seaborn-white')
import math
import pandas as pd
from pyspec.loader import Spectra

import pathlib
from splash import Spectrum, SpectrumType, Splash


class Encoder:
    """
    class to easily encode spectra into a graphical form, to be used for machine learning. The encoded spectra can be automaticall
    persited to a directory, if it has been provided
    """

    def __init__(self, width=512, height=512, min_mz=0, max_mz=2000, plot_axis=False, intensity_max=1000, dpi=72,
                 directory: Optional = None):
        """
            inits the encoder with some standard settings
        """
        self.width = width
        self.height = height
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.axis = plot_axis
        self.intensity_max = intensity_max
        self.dpi = dpi
        self.directory = directory

    @property
    def directory(self):
        return self.__directory

    @directory.setter
    def directory(self, directory):
        self.__directory = directory

        if self.directory is not None:
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    def encode(self, spec: Spectra, prefix: str = None, store_string: bool = False) -> str:
        """
        encodes a spectra to a string in graphical form
        :param spec:
        :param prefix:
        :param store_string:
        :return:
        """
        return self._encode(spec, prefix, store_string)

    def _encode(self, spec: Spectra, prefix: str = None, store_string: bool = False):
        """
        encodes the given spectra
        :param spec: spectra
        :param prefix: prefix
        :param store_string: do you also want to store the spectra string for each spectra?
        :return: an image representation of the encoded spectra in form of a string
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

            plt.tight_layout()
            #        plt.show()
            fig.canvas.draw()
            size = fig.get_size_inches() * fig.dpi

            spectra_string = fig.canvas.tostring_rgb()

            if self.directory is not None:
                name = Splash().splash(Spectrum(spec.spectra, SpectrumType.MS))
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

                directory = self.directory if prefix is None else "{}/{}".format(self.directory, prefix)
                pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
                plt.savefig("{}/{}.png".format(directory, name), dpi=self.dpi)

                plt.close(fig=fig)

                if store_string:
                    with open("{}/{}.txt".format(directory, name), 'w') as the_file:
                        the_file.write(spec.spectra)
                return None

            return spectra_string
        except ValueError:
            pass

    def encodes(self, spectra: List[Spectra]):
        """
        encodes a spectra as picture. Conceptually wise
        we will render 3 dimensions

        x is MZ as nominal
        y is MZ as accurate
        z is intensity between 0 and 100
        :param spectra:
        :return:
        """

        for spec in spectra:
            plt = self.encode(spec)


class DualEncoder(Encoder):
    """
    this encoder encodes the data in form of 2 charts. 1 chart the actual spectra and the other a heatmap of accuracies
    """
