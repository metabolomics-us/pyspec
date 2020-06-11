import requests
from typing import List, Tuple

from pyspec.loader import Spectra


class MoNALoader:
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
        property_names = ['precursor m/z', 'precursor type', 'instrument', 'instrument type', 'ionization mode']
        property_values = [self._find_metadata(spectrum, p) for p in property_names]

        # build spectrum object
        return Spectra(
            id=spectrum['id'],
            spectra=spectrum['spectrum'],
            splash=spectrum['splash']['splash'] if 'splash' in spectrum else None,
            name=name,
            ms_level=ms_level,
            inchiKey=inchikey,
            properties={k: v for k, v in zip(property_names, property_values) if v is not None},
            submitter=spectrum['submitter'],
            library=spectrum.get('library', None)
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
        elif r.status_code == 404:
            return None
        else:
            raise Exception(f'error retriving MoNA spectrum with id: {id}')

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
            raise Exception(f'error when executing query on MoNA: {rsql_query}')

    def similarity_serarch(self, spectrum: str, min_similarity: float = 0.5,
                           precursor_mz: float = None, precursor_tolerance: float = None,
                           required_tags: List[str] = None, filter_tags: List[str] = None,
                           size: int = 20) -> List[Tuple[Spectra, float]]:
        """
        execute a similarity search
        :param spectrum:
        :param min_similarity: minimum similarity score from 0 to 1
        :param precursor_mz: optional precursor
        :param precursor_tolerance: precursor search tolerance in dalton
        :param required_tags: hits must match all of the required tags
        :param filter_tags: hits must match at least one of the filter tags
        :param size: number of search results to return
        :return:
        """

        # build request body
        body = {
            'spectrum': spectrum,
            'minSimilarity': min_similarity
        }

        if precursor_mz is not None:
            body['precursorMZ'] = precursor_mz
        if precursor_tolerance is not None:
            body['precursorToleranceDa'] = precursor_tolerance
        if required_tags is not None:
            body['requiredTags'] = required_tags
        if filter_tags is not None:
            body['filterTags'] = filter_tags


        # build url
        url = f'{self.mona_url}/rest/similarity/search?size={size}'

        r = requests.post(url, json=body)

        # execute query
        if r.status_code == 200:
            data = r.json()
            return [(self._parse_spectrum(x['hit']), x['score']) for x in data]
        else:
            raise Exception(f'error when executing similarity search on MoNA')

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


class MoNAQueryGenerator:
    """
    simple MoNA RSQL query generator
    """

    def query_by_name(self, name: str) -> str:
        return f'compound.names=q=\'name=like="{name}"\''

    def query_by_inchikey(self, inchikey: str) -> str:
        return f'compound.metaData=q=\'name=="InChIKey" and value=="{inchikey}"\''

    def query_by_partial_inchikey(self, partial_inchikey: str) -> str:
        return f'compound.metaData=q=\'name=="InChIKey" and value=match=".*{partial_inchikey}.*"\''

    def query_by_splash(self, splash: str) -> str:
        return f'splash.splash=={splash}'

    def query_by_metadata(self, metadata_name: str, metadata_value: str) -> str:
        return f'metaData=q=\'name=="{metadata_name}" and value=="{metadata_value}"\''

    def query_by_tag(self, tag: str) -> str:
        return f'tags.text=={tag}'

    def query_by_compound_classification(self, classification: str) -> str:
        return f'compound.classification.value=="{classification}"'

    def join_queries(self, queries: List[str]) -> str:
        return ' and '.join(queries)
