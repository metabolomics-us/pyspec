from pymzml.spec import Spectrum as PySpectrum
from splash import SpectrumType, Splash, Spectrum

from pyspec.parser.pymzl.filters import MSMinLevelFilter
from pyspec.parser.pymzl.msms_finder import MSMSFinder


class MZMLtoCSVConverter:
    """
    converts an mszml file to a CSV file, which contains useful metadata for machine learning operations
    """

    def convert(self, input: str, output: str):
        """
        converts the given file to a csv file containing all msms information
        :param name:
        :return:
        """
        finder = MSMSFinder()

        with open(output, "w+") as out:
            out.write("Level;Basepeak;Basepeak Intensity;Time;Splash;MSMS\n")

            def callback(msms: PySpectrum, file_name: str):
                highest = msms.highest_peaks(1)[0]
                spectra = msms.convert(msms).spectra
                splash = Splash().splash(Spectrum(spectra, SpectrumType.MS))
                out.write(
                    "{};{};{};{};{};{}\n".format(msms.ms_level, highest[0], highest[1], msms.scan_time[0], splash,
                                                 spectra))

            finder.locate(msmsSource=input, callback=callback, filters=[MSMinLevelFilter(2)])
