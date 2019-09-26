from operator import itemgetter

import psycopg2
from peewee import DoesNotExist
from pymzml.spec import Spectrum as PySpectrum
from splash import SpectrumType, Splash, Spectrum
from tqdm import tqdm

from pyspec import config
from pyspec.converter.dataset_to_postgres import DatesetToPostgresConverter
from pyspec.machine.persistence.model import MZMLSampleRecord, MZMLMSMSSpectraRecord, db, \
    MZMZMSMSSpectraClassificationRecord
from pyspec.parser.pymzl.filters import MSMinLevelFilter
from pyspec.parser.pymzl.msms_finder import MSMSFinder


class BinBasetoPostgresConverter:
    """
    converts all visibile binbase data to the postgres database
    """

    def __init__(self):
        db.create_tables([MZMLSampleRecord, MZMLMSMSSpectraRecord, MZMZMSMSSpectraClassificationRecord])

    def convert(self, pattern: str = "%%"):
        """
        converts the given input and stores it at the defined postgres database location
        :param input:
        :return:
        """

        # 1. query all visble sample information
        db = config.config(filename="database.ini", section="binbase")

        connection = psycopg2.connect(**db)

        cursor = connection.cursor()
        sample_count = connection.cursor()
        spectra = connection.cursor()
        bin = connection.cursor()

        sample_count.execute(
            "select count(*) from samples where sample_name like '{}' and visible = 'TRUE'".format(pattern))
        count = sample_count.fetchone()[0]

        pbar = tqdm(total=count + 1, desc="importing samples for pattern {}".format(pattern))

        cursor.execute(
            "select sample_id, sample_name from samples where sample_name like '{}' and visible = 'TRUE'".format(
                pattern))

        row = cursor.fetchone()

        while row is not None:
            try:
                try:
                    record = MZMLSampleRecord.get(MZMLSampleRecord.file_name == row[1])
                    record.delete_instance()
                except Exception:
                    # object doesn't exist
                    pass
                # 2. create sample object
                MZMLSampleRecord.create(file_name=row[1], instrument="gctof", name=row[1])

                record = MZMLSampleRecord.get(MZMLSampleRecord.file_name == row[1])
                spectra.execute(
                    "select bin_id, spectra_id,spectra, retention_time from spectra where sample_id = {}".format(
                        row[0]))

                s = spectra.fetchone()

                while s is not None:
                    splash = Splash().splash(Spectrum(s[2], SpectrumType.MS))

                    spectrum = [list(map(float, x.split(':'))) for x in s[2].strip().split()]

                    spectrum_max = max(spectrum, key=itemgetter(1))

                    MZMLMSMSSpectraRecord.create(sample=record, msms=s[2], rt=s[3],
                                                 splash=splash,
                                                 level=1, base_peak=spectrum_max[0],
                                                 base_peak_intensity=spectrum_max[1],
                                                 precursor=0,
                                                 precursor_intensity=0,
                                                 precursor_charge=0,
                                                 ion_count=len(spectrum))

                    if s[0] is not None:
                        DatesetToPostgresConverter.classify("bin_id", splash, s[0])
                        bin.execute("select name from bin where bin_id = {} and bin_id::text != name".format(s[0]))
                        result = bin.fetchone()
                        if result is not None:
                            DatesetToPostgresConverter.classify("bin", splash, result[0])

                    s = spectra.fetchone()

                row = cursor.fetchone()
            finally:
                pbar.update(1)
