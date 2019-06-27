import os
from abc import abstractmethod

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

    def generate_dataframe(self, input) -> DataFrame:
        """
        generates a dataframe for the given input with all the internal labels
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

    def generate_labels(self, input: str, callback):
        data = "{}/train".format(input)

        for category in os.listdir(data):
            for file in os.listdir("{}/{}".format(data, category)):
                callback(file, category)
