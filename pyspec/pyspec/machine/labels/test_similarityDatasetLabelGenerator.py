import tabulate

from pyspec.machine.labels.similarity_labels import SimilarityDatasetLabelGenerator, \
    EnhancedSimilarityDatasetLabelGenerator


def test_generate_labels():
    generator = SimilarityDatasetLabelGenerator(spectra_per_compounds=500)

    result = generator.generate_dataframe("not important")

    assert result is not None
    print(tabulate.tabulate(result[0], headers='keys'))


def test_generate_labels_enhanced():
    generator = EnhancedSimilarityDatasetLabelGenerator(limit=500)

    result = generator.generate_dataframe("not important")

    assert result is not None
    print(tabulate.tabulate(result[0], headers='keys'))

