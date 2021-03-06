import argparse
import csv
import traceback
from typing import List

from pyspec.loader import Spectra
from pyspec.loader.binvestigate import BinVestigate

from tqdm import tqdm

from pyspec.msp.writer import MSP


class ParseReport:
    """
    parses a binvestigate report, based on the following format

    species organ   binid   bin count

    and generates binvestigate spectra from it, which can be saved as MSP file

    """

    _cache = {}

    def parse_file(self, file: str, delimiter=",", known_only=False) -> List[Spectra]:
        """
        parses a file
        :param file:
        :return:
        """

        result = []
        with open(file, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar='"', quoting=csv.QUOTE_ALL, skipinitialspace=True)

            counter = 0
            for row in tqdm(reader, "reading data file..."):

                if counter == 0:
                    counter = counter + 1
                else:
                    try:
                        spectra = self.parse_string(row)

                        if known_only:
                            if spectra.inchiKey is not None:
                                result.append(spectra)
                        else:
                            result.append(spectra)
                    except Exception as e:
                        traceback.print_exc()

        return result

    def parse_string(self, data: str) -> Spectra:
        """
        parses the given string and returns a list of spectra
        :param file:
        :return:
        """

        assert len(data) == 5, f"incoming data needs to be a list of 5, it was of len {len(data)}" \
            f" and content was {data}"

        try:
            spectra = self.load_spectra(int(data[2]))
            return Spectra(
                name=data[3],
                spectra=spectra.spectra,
                inchiKey=spectra.inchiKey,
                splash=spectra.splash,
                properties={
                    "organ": data[1],
                    "species": data[0],
                    "annotations": data[4]
                }
            )
        except Exception as e:
            raise Exception(f"error observed at string to parse: {data}")

    def load_spectra(self, id: int) -> Spectra:
        """
        loads a spectra from the server
        :param id:
        :return:
        """
        if id not in self._cache:
            self._cache[id] = BinVestigate().load(id)

        return self._cache[id]


def main():
    parser = argparse.ArgumentParser(description='Generate MSP from BinVestigate export.')
    parser.add_argument("input", type=str, help="please provide a correctly formatted input file")
    parser.add_argument("output", type=str, help="where would you like to store the result")
    parser.add_argument("--top", type=int, help="limit output to the top-n reported bin's by count", default=0)

    args = parser.parse_args()

    report = ParseReport()
    parsed = report.parse_file(args.input, known_only=True)

    if args.top > 0:
        parsed = sorted(parsed, key=lambda x: x.properties['annotations'], reverse=True)
        parsed = parsed[:args.top]

    msp = MSP()

    with open(args.output, 'w') as out:
        for x in tqdm(parsed, "writing data out"):
            out.write(msp.from_spectra(x))
            out.write("\n")


if __name__ == "__main__": main()
