import os

import pytest
import tabulate

from pyspec.machine.labels.generate_labels import DirectoryLabelGenerator, CSVLabelGenerator, MachineDBDataSetGenerator

generators = [DirectoryLabelGenerator(), CSVLabelGenerator(), MachineDBDataSetGenerator()]

datasets = [("clean_dirty", 88, 13, 3), ("pos_neg", 2543, 14, 3)]
folder = "datasets"


@pytest.mark.parametrize("generator", generators)
@pytest.mark.parametrize("dataset", datasets)
def test_generate_dataframe(generator, dataset):
    results = generator.generate_dataframe("{}/{}".format(folder, dataset[0]))

    #    print(tabulate.tabulate(results[0], headers='keys'))
    assert len(results) == 2
    # ensure all reported files exist
    assert results[0].shape == (dataset[1], dataset[3])

    # check the size of the loaded test data if applies
    if generator.contains_test_data():
        assert results[1].shape == (dataset[2], dataset[3])
