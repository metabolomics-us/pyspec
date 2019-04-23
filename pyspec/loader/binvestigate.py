import requests

from pyspec.loader import Spectra


class BinVestigate:
    """
    downloads a given bin from binvestigate
    whith it's properties
    """

    def load(self, id: int):
        data = requests.get(f"https://binvestigate.fiehnlab.ucdavis.edu/rest/bin/{id}")

        if data.status_code != 200:
            raise Exception(f"sorry there was an error with this requests for id: {id}")

        data = data.json()
        return Spectra(
            spectra=data['spectra'],
            name=data['name'],
            ms_level=1
        )
