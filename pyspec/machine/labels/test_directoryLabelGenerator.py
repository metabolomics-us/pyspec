from unittest import TestCase

from pyspec.machine.labels.generate_labels import DirectoryLabelGenerator


def test_generate_labels():
    dataset = "clean_dirty"
    folder = "datasets"

    generator = DirectoryLabelGenerator()

    result = generator.generate_dataframe("{}/{}".format(folder, dataset))

    assert result.shape == (176, 2)

    generator.to_csv("{}/{}".format(folder, dataset), "{}/{}/train.csv".format(folder, dataset))
