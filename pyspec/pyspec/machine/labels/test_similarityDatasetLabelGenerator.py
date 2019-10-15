import tabulate

from pyspec.machine.labels.generate_labels import SimilarityDatasetLabelGenerator


def test_generate_labels():
    generator = SimilarityDatasetLabelGenerator(limit=500)

    result = generator.generate_dataframe("not important")

    assert result is not None
    print(tabulate.tabulate(result[0], headers='keys'))
