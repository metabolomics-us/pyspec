import os

import math
import pymzml
import requests
from tqdm import tqdm


class MSMSFinder:
    """
    finds all the MSMS spectra in the given file and if it's an url, download it first
    """

    def __init__(self):
        """

        """

    def locate(self, msmsSource: str, callback):
        """
            loads the given source and invokes the callback for each spectra found
        :param msmsSource: needs to be a URL of some kind, file:// http:// you figure it out
        :param callback: callback, to work on all the parse MSMS spectra and do some magic with them
        :return:
        """

        file_name = self.download_rawdata(msmsSource, "data")

        reader = pymzml.run.Reader(file_name)

        for spectra in tqdm(reader, total=reader.get_spectrum_count(),
                            unit='spectra',
                            unit_scale=True, leave=True, desc=f"analyzing spectra in {file_name}"):

            # we only care about msms spectra
            if spectra.ms_level > 1:
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
