import os
import traceback
from glob import iglob

from peewee import DoesNotExist

from pyspec.machine.persistence.model import db, MZMLSampleRecord, MZMLMSMSSpectraRecord, \
    MZMZMSMSSpectraClassificationRecord


class DatesetToPostgresConverter:
    """
    converts existing datasets to the postgres database
    """

    def __init__(self):
        db.create_tables([MZMLSampleRecord, MZMLMSMSSpectraRecord, MZMZMSMSSpectraClassificationRecord])

    def convert_clean_dirty(self, dataset: str, folder: str = "datasets"):
        """
        converts a clean dirty dataset to a postgres dataset, assuming the
        linked splashes exist in the database already. Otherwise they will be skipped
        :param dataset: 
        :return: 
        """
        file = "{}/{}".format(folder, dataset)
        test = "{}/{}".format(file, "test")
        train = "{}/{}".format(file, "train")

        self._generate_clean_dirt_record(test)
        self._generate_clean_dirt_record(train)

    def _generate_clean_dirt_record(self, dir: str):
        with db.atomic():
            for category in os.listdir(dir):
                for file in iglob("{}/{}/**/*.png".format(dir, category), recursive=True):
                    name = file.split("/")[-1].split(".")[0]

                    try:
                        spectra = MZMLMSMSSpectraRecord.get(MZMLMSMSSpectraRecord.splash == name)
                        MZMZMSMSSpectraClassificationRecord.create(spectra=spectra, category=category)
                    except DoesNotExist as e:
                        pass
