from unittest import TestCase

from pyspec.machine.labels.generate_labels import CSVLabelGenerator


def test_generate_labels():
    dataset = "clean_dirty"
    folder = "datasets"

    generator = CSVLabelGenerator()

    result = generator.generate_dataframe("{}/{}/train.csv".format(folder, dataset))

    print(result)

    assert result.shape == (176,2)