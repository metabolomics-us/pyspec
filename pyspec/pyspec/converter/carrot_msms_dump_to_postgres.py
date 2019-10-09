import gzip
import json

from tqdm import tqdm

from pyspec.converter.dataset_to_postgres import DatesetToPostgresConverter
from pyspec.machine.persistence.model import db, MZMLSampleRecord, MZMLMSMSSpectraRecord, \
    MZMZMSMSSpectraClassificationRecord


class LCBinBaseToPostgresConverter:
    """
    converts a lc binbase msms dump to the postgres machine database.
    """

    def __init__(self):
        db.create_tables([MZMLSampleRecord, MZMLMSMSSpectraRecord, MZMZMSMSSpectraClassificationRecord])

    def convert(self, file, compressed: bool = False) -> int:
        """
        converts all dumped msms json files into the local postgres database, assuming
        the rawdata sample is stored in the database
        :param dataset:
        :return: 
        """
        count = 0

        if compressed is True:
            with gzip.open(file, 'r') as fin:
                data = json.load(fin)
        else:
            with open(file) as f:
                data = json.load(f)

        if 'spectra' in data:
            for x in tqdm(data['spectra'], desc="importing file {}".format(file)):
                if x['name'] != 'Unknown':
                    DatesetToPostgresConverter.classify(category="origin", splash=x['raw splash'], value="lc-binbase")
                    DatesetToPostgresConverter.classify(category="name", splash=x['raw splash'], value=x['name'])
                    count = count + 1

        return count
