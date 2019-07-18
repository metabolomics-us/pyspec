import os

import pytest

from pyspec.machine.labels.generate_labels import DirectoryLabelGenerator, CSVLabelGenerator

generators = [DirectoryLabelGenerator(), CSVLabelGenerator()]

datasets = [("clean_dirty", 88, 13), ("pos_neg", 2543, 14)]
folder = "datasets"


@pytest.mark.parametrize("generator", generators)
@pytest.mark.parametrize("dataset", datasets)
def test_generate_dataframe(generator, dataset):
    result = generator.generate_dataframe("{}/{}".format(folder, dataset[0]))

    # ensure all reported files exist
    assert result['file'].apply(lambda x: os.path.exists(x)).all()
    assert result.shape == (dataset[1], 2)
    generator.to_csv("{}/{}".format(folder, dataset[0]), "{}/{}/train.csv".format(folder, dataset[0]))


@pytest.mark.parametrize("generator", generators)
@pytest.mark.parametrize("dataset", datasets)
def test_generate_test_dataframe(generator, dataset):
    result = generator.generate_test_dataframe("{}/{}".format(folder, dataset[0]))
    assert result['file'].apply(lambda x: os.path.exists(x)).all()
    assert result.shape == (dataset[2], 1)

    import pandas as pd
    pd.set_option('display.max_columns', 30)
    pd.set_option("display.max_colwidth", 10000)

