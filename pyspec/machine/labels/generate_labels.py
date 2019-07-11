import csv
import os
from abc import abstractmethod
from glob import iglob

from pandas import DataFrame


class LabelGenerator:
    """
    class to easily generate a file for us containing all the labels
    this is based on pictures in directories


    """

    @abstractmethod
    def generate_labels(self, input: str, callback):
        """
        :param input: the input file to utilize
        :param callback: def callback(identifier, class)
        :return:
        """

    @abstractmethod
    def generate_test_dataframe(self, input: str, abs: bool = False) -> DataFrame:
        """
        generates a test dataframe for us
        :param input:
        :return:
        """

    def generate_dataframe(self, input: str) -> DataFrame:
        """
        generates a dataframe for the given input with all the internal labels. This will be used for training and validation
        :param input:
        :return:
        """
        data = []

        def callback(id, category):
            nonlocal data
            data.append({
                "file": id,
                "class": category
            })

        self.generate_labels(input, callback)

        return DataFrame(data)

    def to_csv(self, input: str, file_name: str):
        """
        reads all the images, and saves them as a CSV file
        :param input: from where to load the data
        :param file_name: name of the labeled datafile
        :return:
        """
        result = self.generate_dataframe(input)
        result.to_csv(file_name, encoding='utf-8', index=False)


class DirectoryLabelGenerator(LabelGenerator):
    """

    generates labels from pictures in a directory
    , which needs to be configured like this

    dataset_name/train/class
    dataset_name/test/class

    for example

    dataset_spectra/train/clean
    dataset_spectra/train/dirty
    dataset_spectra/test/clean
    dataset_spectra/test/dirty

    """

    def generate_test_dataframe(self, input: str, abs: bool = False) -> DataFrame:
        data = "{}/test".format(input)
        result = []

        for category in os.listdir(data):
            for file in iglob("{}/{}/**/*.png".format(data, category), recursive=True):

                if not abs:
                    result.append({
                        "file": file,
                    })
                else:
                    result.append({
                        "file": os.path.abspath(file),
                    })

        return DataFrame(result)

    def generate_labels(self, input: str, callback):
        data = "{}/train".format(input)

        for category in os.listdir(data):
            for file in iglob("{}/{}/**/*.png".format(data, category), recursive=True):
                callback(file, category)


class CSVLabelGenerator(LabelGenerator):
    """
    generates labels from a CSV file
    """

    def generate_test_dataframe(self, input: str, abs: bool = False) -> DataFrame:
        import os
        assert os.path.exists(input), "please ensure that {} exists!".format(input)
        csv_file = os.path.join(input, "test.csv")
        assert os.path.isfile(csv_file), "please ensure that {} is a file!".format(csv_file)

        with open(csv_file, mode='r') as infile:
            reader = csv.reader(infile)

            # first row is headers

            row = next(reader)

            data = []
            assert len(row) == 1, "please ensure you have exactly 1 column!"
            for row in reader:
                if len(row) == 1:
                    if os.path.exists(row[0]):
                        data.append({'file': row[0]})
                    elif os.path.exists("{}/{}".format(input, row[0])):
                        data.append({'file': "{}/{}".format(input,row[0])})
                    else:
                        raise Exception("sorry we did not find the file: {} or {}/{}".format(row[0], input, row[0]))

            return DataFrame(data)

    def generate_labels(self, input: str, callback):
        import os
        assert os.path.exists(input), "please ensure that {} exists!".format(input)
        input = os.path.join(input, "train.csv")
        assert os.path.isfile(input), "please ensure that {} is a file!".format(input)

        with open(input, mode='r') as infile:
            reader = csv.reader(infile)

            # first row is headers

            row = next(reader)

            assert len(row) == 2, "please ensure you have exactly 2 columes!"

            if row[0] == self.field_category:
                c = 0
                f = 1
            elif row[1] == self.field_category:
                c = 1
                f = 0
            else:
                assert False, "please ensure that your column names are {} and {} instead of {}".format(
                    self.field_category, self.field_id, row)

            for row in reader:
                callback(row[f], row[c])

    def __init__(self, field_id: str = "file", field_category: str = "class"):
        self.field_id = field_id
        self.field_category = field_category
