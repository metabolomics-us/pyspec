import os
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

    def convert_dataset(self, dataset: str, folder: str = "datasets") -> int:
        """
        converts a dataset to a postgres dataset, assuming the
        linked splashes exist in the database already. Otherwise they will be skipped
        :param dataset: 
        :return: 
        """
        file = "{}/{}".format(folder, dataset)
        test = "{}/{}".format(file, "test")
        train = "{}/{}".format(file, "train")

        count = self._generate_classification_record(test)
        count += self._generate_classification_record(train)

        return count

    def _generate_classification_record(self, dir: str) -> int:
        """
        generates a new classification record
        :param dir: 
        :return: 
        """

        count = 0

        with db.atomic():
            for category in os.listdir(dir):
                for file in iglob("{}/{}/**/*.png".format(dir, category), recursive=True):
                    name = file.split("/")[-1].split(".")[0]

                    try:
                        spectra = MZMLMSMSSpectraRecord.get(MZMLMSMSSpectraRecord.splash == name)

                        try:
                            MZMZMSMSSpectraClassificationRecord.get(
                                MZMZMSMSSpectraClassificationRecord.spectra == spectra,
                                MZMZMSMSSpectraClassificationRecord.category == category).delete()
                        except DoesNotExist as e:
                            pass

                        MZMZMSMSSpectraClassificationRecord.replace(spectra=spectra, category=category)
                        count += 1
                    except DoesNotExist as e:
                        pass

        return count
