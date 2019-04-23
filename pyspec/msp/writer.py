from pyspec.loader import Spectra
from pyspec.loader.binvestigate import BinVestigate


class MSP:
    """
    helper class to quickly generate a MSP file from a formated spectra string.

    format is to be supposed:

    ion:intensity ion:intensity
    """

    def from_bin(self, id: int) -> str:
        return self.from_spectra(BinVestigate().load(id))

    def from_spectra(self, spectra: Spectra) -> str:
        return self.to(spectra=spectra.spectra, name=spectra.name, properties={"Ms Level": spectra.ms_level})

    def from_str(self, spectra: str, name: str, properties: dict) -> str:
        """
        generates a MSP representation of the incomming data
        :param spectra:
        :param name:
        :param properties:
        :return:
        """

        msp = []
        msp.append("Name: {}".format(name))
        for k, v in properties:
            msp.append("{}: {}".format(k, v))

        msp.append('Num Peaks: %d' % len(spectra))
        msp.extend(' '.join(x.split(':')) for x in spectra.split())

        return '\n'.join(msp)
