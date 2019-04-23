class MSP:
    """
    helper class to quickly generate a MSP file from a formated spectra string.

    format is to be supposed:

    ion:intensity ion:intensity
    """

    def to(self, spectra: str, name: str, properties: dict) -> str:
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
