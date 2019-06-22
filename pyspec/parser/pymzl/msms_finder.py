import os

from pymzml.spec import Spectrum
from typing import List, Optional

import math
import pymzml
import requests
from tqdm import tqdm

from pyspec.parser.pymzl.filters import Filter
from pyspec.loader import Spectra


class MSMSFinder:
    """
    finds all the MSMS spectra in the given file and if it's an url, download it first
    """

    def toSpectra(self, spectra: Spectrum) -> Spectra:
        peaks = spectra.peaks("centroided")

        f = lambda x: "{}:{}".format(x[0], x[1])
        result = []
        for x in peaks:
            result.append(f(x))

        result = " ".join(result)
        return Spectra(
            spectra=result
        )

    def __init__(self):
        """

        """

    def locate(self, msmsSource: str, callback, filters: Optional[List[Filter]] = None):
        """
            loads the given source and invokes the callback for each spectra found
        :param msmsSource: needs to be a URL of some kind, file:// http:// you figure it out
        :param callback: callback, to work on all the parse MSMS spectra and do some magic with them
        :param filters: a list of optional filters to be applied, to reduce processing overhead
        :return:
        """

        file_name = self.download_rawdata(msmsSource, "data")

        reader = pymzml.run.Reader(file_name)

        for spectra in tqdm(reader, total=reader.get_spectrum_count(),
                            unit='spectra',
                            unit_scale=True, leave=True, desc=f"analyzing spectra in {file_name}"):

            spectra.convert = self.toSpectra

            if filters is not None:
                for x in filters:
                    if x.accept(spectra):
                        callback(spectra, file_name)
            else:
                callback(spectra, file_name)

    def download_rawdata(self, source, dir: str = "data"):
        """
        downloads a rawdata file and stores it in the local directory
        :param source:
        :param dir where to store the downloaded data
        :return:
        """

        if os.path.exists(source):
            # not a url
            return source

        file_name = os.path.basename(source)
        response = requests.get(source, stream=True)
        file_name = f"{dir}/{file_name}"

        if not os.path.exists(os.path.dirname(file_name)):
            os.mkdir(dir)
        if not os.path.exists(file_name):
            block_size = 1024
            total_size = int(response.headers.get('content-length', 0))

            with open(file_name, "wb") as file:
                for data in tqdm(response.iter_content(block_size), total=math.ceil(total_size // block_size),
                                 unit='KB',
                                 unit_scale=True, leave=True, desc=f"downloading raw data to {file_name}"):
                    file.write(data)

        return file_name
