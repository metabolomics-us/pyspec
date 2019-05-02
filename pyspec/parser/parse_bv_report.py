from typing import List

from pyspec.loader import Spectra
from pyspec.loader.binvestigate import BinVestigate


class ParseReport:
    """
    parses a binvestigate report, based on the following format

    species organ   binid   bin count

    and generates binvestigate spectra from it, which can be saved as MSP file

    """

    def parse_file(self, file: str) -> List[Spectra]:
        """
        parses a file
        :param file:
        :return:
        """
        with open(file, 'r') as f:
            return self.parse_string(f.readlines())

    def parse_string(self, fileContent: List[str]) -> List[Spectra]:
        """
        parses the given string and returns a list of spectra
        :param file:
        :return:
        """

        result = []

        counter = 0
        for x in fileContent:
            if counter == 0:
                pass
            else:
                data = x.rstrip('\n').split("\t")

                result.append(Spectra(
                    name=data[3],
                    spectra=self.load_spectra(int(data[2])),
                    properties={
                        "organ": data[1],
                        "species": data[0],
                        "annotations": data[4]
                    }

                )
                )
            counter = counter + 1

        return result

    def load_spectra(self, id: int) -> str:
        """
        loads a spectra from the server
        :param id:
        :return:
        """
        return BinVestigate().load(id).spectra
