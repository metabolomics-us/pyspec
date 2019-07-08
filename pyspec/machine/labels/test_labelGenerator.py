import pytest

from pyspec.machine.labels.generate_labels import DirectoryLabelGenerator, CSVLabelGenerator

generators = [DirectoryLabelGenerator(), CSVLabelGenerator()]

dataset = "clean_dirty"
folder = "datasets"


@pytest.mark.parametrize("generator", generators)
def test_generate_dataframe(generator):
    result = generator.generate_dataframe("{}/{}".format(folder, dataset))
    assert result.shape == (88, 2)
    generator.to_csv("{}/{}".format(folder, dataset), "{}/{}/train.csv".format(folder, dataset))


@pytest.mark.parametrize("generator", generators)
def test_generate_test_dataframe(generator):
    result = generator.generate_test_dataframe("{}/{}".format(folder, dataset))
    print(result)
    assert result.shape == (13, 1)

    import pandas as pd
    pd.set_option('display.max_columns', 30)
    pd.set_option("display.max_colwidth", 10000)

