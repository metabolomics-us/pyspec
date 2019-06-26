from unittest import TestCase

from pyspec.machine.labels.generate_labels import DirectoryLabelGenerator


def test_generate_labels():
    dataset = "clean_dirty"
    folder = "datasets"

    generator = DirectoryLabelGenerator()

    result = generator.generate_dataframe("{}/{}".format(folder, dataset))

    print(result)
