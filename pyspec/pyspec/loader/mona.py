import requests
from typing import List

from pyspec.loader import Spectra


class MoNA:
    """
    simple MoNA client
    """

    def __init__(self):
        self.mona_url = 'https://mona.fiehnlab.ucdavis.edu'


    def _find_metadata(self, spectrum: dict, name: str, compound: bool = False):
        """
        search for a given metadata name in the spectrum or compound metadata and return the first match or None
        :param spectrum:
        :param name:
        :param compound:
        :return:
        """

        # get metadata list
        if compound:
            if len(spectrum['compound']) > 0:
                metaData = spectrum['compound'][0]['metaData']
            else:
                metaData = []
        else:
            metaData = spectrum['metaData']

        # find metadata value
        values = [m['value'] for m in metaData if m['name'] == name]

        if len(values) > 0:
            return values[0]
        else:
            return None

    def _parse_spectrum(self, spectrum: dict) -> Spectra:
        """
        convert a raw MoNA JSON object to a Spectra object
        :param spectrum:
        :return:
        """

        # get ms level
        ms_level = str(self._find_metadata(spectrum, 'ms level'))

        if ms_level[-1].isnumeric():
            ms_level = int(ms_level[-1])
        else:
            ms_level = -1

        # get compound properties
        inchikey = self._find_metadata(spectrum, 'InChIKey', compound=True)

        if len(spectrum['compound']) > 0 and len(spectrum['compound'][0]['names']) > 0:
            name = spectrum['compound'][0]['names'][0]['name']
        else:
            name = spectrum['id']

        # get additional properties
        property_names = ['precursor m/z', 'precursor type', 'instrument', 'instrument type']
        property_values = [self._find_metadata(spectrum, p) for p in property_names]

        # build spectrum object
        return Spectra(
            spectrum=spectrum['spectrum'],
            splash=spectrum['splash']['splash'] if 'splash' in spectrum else None,
            name=name,
            ms_level=ms_level,
            inchiKey=inchikey,
            properties={k: v for k, v in zip(property_names, property_values) if v is not None}
        )


    def load_spectrum(self, id: str) -> Spectra:
        """
        retrieve a spectrum by its id
        :param id:
        :return:
        """

        r = requests.get(f'{self.mona_url}/rest/spectra/{id}')

        if r.status_code == 200:
            return self._parse_spectrum(r.json())
        else:
            raise Exception(f'no spectrum available on MoNA with id: {id}')

    def query(self, rsql_query: str, page: int = None, page_size: int = None) -> List[Spectra]:
        """
        execute a query with optional page properties
        :param rsql_query:
        :param page:
        :param page_size:
        :return:
        """

        # build url
        url = f'{self.mona_url}/rest/spectra/search?'

        if page is not None:
            url += f'page={page}&'
        if page_size is not None:
            url += f'size={page_size}&'

        # execute query
        r = requests.get(f'{url}query={rsql_query}')

        if r.status_code == 200:
            data = r.json()
            return [self._parse_spectrum(s) for s in data]
        else:
            raise Exception(f'no spectra found on MoNA for query: {rsql_query}')

    def list_metadata_names(self) -> List[str]:
        """
        retrieve a list of all metadata names
        :return:
        """

        r = requests.get(f'{self.mona_url}/rest/metaData/names')

        if r.status_code == 200:
            data = r.json()
            return [x['name'] for x in data]
        else:
            raise Exception(f'no metadata names available')

    def list_metadata_values(self, metadata_name: str):
        """
        retrieve a list of metadata values associated with the given metadata name
        :param metadata_name:
        :return:
        """

        r = requests.get(f'{self.mona_url}/rest/metaData/values?name={metadata_name}')

        if r.status_code == 200:
            data = r.json()
            return [x['value'] for x in data['values']]
        else:
            raise Exception(f'no metadata values available for: {metadata_name}')

