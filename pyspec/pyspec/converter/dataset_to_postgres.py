import os
import traceback
from glob import iglob

from peewee import DoesNotExist
from tqdm import tqdm

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

        count = self._generate_classification_record(train, dataset)
        count = self._generate_classification_record(test, dataset) + count

        return count

    def _generate_classification_record(self, dir: str, category: str) -> int:
        """
        generates a new classification record
        :param dir: 
        :return: 
        """

        count = 0

        with db.atomic():
            for value in os.listdir(dir):
                for file in tqdm(iglob("{}/{}/**/*.png".format(dir, value), recursive=True),
                                 desc="converting dataset path {}/{}".format(dir, value)):
                    name = file.split("/")[-1].split(".")[0]

                    try:
                        self.classify(category, name, value)
                        count += 1
                    except DoesNotExist as e:
                        # print("splash not found!")
                        pass

        return count

    @staticmethod
    def classify(category: str, splash: str, value: str):
        """
        stores a classification record or overwrites it
        :param category:
        :param splash: linked MSMS spectra, yes it's not 100% unique...
        :param value:
        :return:
        """
        # print("looking for {}".format(splash))
        spectra = MZMLMSMSSpectraRecord.get(MZMLMSMSSpectraRecord.splash == splash)
        # print(spectra)
        try:
            deleted = MZMZMSMSSpectraClassificationRecord.get(
                MZMZMSMSSpectraClassificationRecord.spectra == spectra,
                MZMZMSMSSpectraClassificationRecord.category == category,
                MZMZMSMSSpectraClassificationRecord.predicted == False
            ).delete_instance()
        except DoesNotExist as e:
            pass
        MZMZMSMSSpectraClassificationRecord.create(spectra=spectra, category=category, value=value, predicted=False)
