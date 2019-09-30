import os

import pytest

from pyspec.machine.labels.generate_labels import DirectoryLabelGenerator, CSVLabelGenerator

generators = [DirectoryLabelGenerator(), CSVLabelGenerator()]

datasets = [("clean_dirty", 88, 13), ("pos_neg", 2543, 14)]
folder = "datasets"


@pytest.mark.parametrize("generator", generators)
@pytest.mark.parametrize("dataset", datasets)
def test_generate_dataframe(generator, dataset):
    results = generator.generate_dataframe("{}/{}".format(folder, dataset[0]))

    assert len(results) == 2
    # ensure all reported files exist
    assert results[0]['file'].apply(lambda x: os.path.exists(x)).all()
    assert results[0].shape == (dataset[1], 3)
    generator.to_csv("{}/{}".format(folder, dataset[0]), "{}/{}/train.csv".format(folder, dataset[0]),training=True)

    assert results[1]['file'].apply(lambda x: os.path.exists(x)).all()
    assert results[1].shape == (dataset[1], 3)
