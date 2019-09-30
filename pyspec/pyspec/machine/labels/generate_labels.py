import csv
import os
from abc import abstractmethod
from glob import iglob
from typing import Tuple

from pandas import DataFrame, read_sql, read_sql_query

from pyspec.machine.persistence.model import db, MZMLSampleRecord, MZMLMSMSSpectraRecord, \
    MZMZMSMSSpectraClassificationRecord


class LabelGenerator:
    """
    class to easily generate a file for us containing all the labels
    this is based on pictures in directories


    """

    @abstractmethod
    def generate_labels(self, input: str, callback, training: bool):
        """
        :param input: the input file to utilize
        :param callback: def callback(identifier, class)
        :return:
        """

    def generate_dataframe(self, input: str) -> Tuple[DataFrame, DataFrame]:
        """
        generates a dataframe for the given input with all the internal labels. This will be used for training and validation
        :param input:
        :return:
        """
        data = []

        def callback(id, category, training: bool):
            nonlocal data

            if self.is_file_based():
                assert os.path.exists(id), 'please ensure all files exist. Missing {}'.format(id)

            data.append({
                "file": id,
                "class": category,
                "training": training
            })

        self.generate_labels(input, callback, training=True)
        self.generate_labels(input, callback, training=False)

        training = DataFrame(list(filter(lambda x: x['training'] is True, data)))
        testing = DataFrame(list(filter(lambda x: x['training'] is False, data)))

        return training, testing

    def returns_multiple(self):
        "do we return multiple fields for our 'file' column and the model needs to flatten it"
        return False

    def is_file_based(self) -> bool:
        """
        if this generator is based on files
        :return:
        """
        return True

    def to_csv(self, input: str, file_name: str, training: bool):
        """
        reads all the images, and saves them as a CSV file
        :param input: from where to load the data
        :param file_name: name of the labeled datafile
        :return:
        """
        result = self.generate_dataframe(input)

        if training is True:
            result[0].to_csv(file_name, encoding='utf-8', index=False)
        else:
            result[1].to_csv(file_name, encoding='utf-8', index=False)


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

    def generate_labels(self, input: str, callback, training: bool):

        data = "{}/train".format(input) if training else "{}/test".format(input)

        for category in os.listdir(data):
            for file in iglob("{}/{}/**/*.png".format(data, category), recursive=True):
                callback(file, category, training)


class CSVLabelGenerator(LabelGenerator):
    """
    generates labels from a CSV file
    """

    def generate_labels(self, input: str, callback, training: bool):
        import os
        assert os.path.exists(input), "please ensure that {} exists!".format(input)
        input_file = os.path.join(input, "train.csv") if training else os.path.join(input, "test.csv")
        assert os.path.isfile(input_file), "please ensure that {} is a file!".format(input_file)

        print("using: {}".format(input_file))
        with open(input_file, mode='r') as infile:
            reader = csv.reader(infile)

            # first row is headers

            row = next(reader)

            assert len(row) >= 2, "please ensure you have more than 2 columns!, But given where {}, '{}'".format(
                len(row), row)

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
                if os.path.exists(row[f]):
                    file = row[f]
                elif os.path.exists("{}/{}".format(input, row[f])):
                    file = "{}/{}".format(input, row[f])
                else:
                    raise Exception("sorry we did not find the file: {} or {}/{}".format(row[f], input, row[f]))

                callback(file, row[c], training)

    def __init__(self, field_id: str = "file", field_category: str = "class"):
        self.field_id = field_id
        self.field_category = field_category


class MachineDBDataSetGenerator(LabelGenerator):
    """
    generates a dataset based on the PeeWee domain classes. Query is done direcly as SQL to reduce overhead
    """

    def __init__(self, fields=['msms']):
        """

        :param fields: which fields you want to select and map. If more than one, the result fieled in the dataframe, will be a list!
        """

        db.create_tables([MZMLSampleRecord, MZMLMSMSSpectraRecord, MZMZMSMSSpectraClassificationRecord])
        self.fields = fields

    def generate_labels(self, input: str, callback, training: bool):
        """
        input is the name of dataset, example 'clean_dirty' which translates to the column 'category' in the database
        :param input:
        :param callback:
        :param training:
        :return:
        """
        input = input.split("/")[-1]

        result = read_sql_query(
            "select a.*, b.value as class from  mzmlmsmsspectrarecord a, mzmzmsmsspectraclassificationrecord b where a.id = b.spectra_id and category = '{}'".format(
                input),
            db.connection())

        for index, row in result.iterrows():
            data = []

            for y in self.fields:
                data.append(row[y])

            if len(data) > 1:
                callback(
                    id=data,
                    category=row['class'],
                    training=training
                )
            else:
                callback(
                    id=data[0],
                    category=row['class'],
                    training=training
                )

    def returns_multiple(self):
        return len(self.fields) > 1

    def is_file_based(self) -> bool:
        return False
