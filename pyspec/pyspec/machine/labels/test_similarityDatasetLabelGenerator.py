import tabulate

from pyspec.machine.labels.similarity_labels import SimilarityDatasetLabelGenerator, \
    EnhancedSimilarityDatasetLabelGenerator


def test_generate_labels():
    generator = SimilarityDatasetLabelGenerator(spectra_per_compounds=2, compound_limit=10)

    result = generator.generate_dataframe("not important")

    assert result is not None
    assert result[0] is not None
    print(tabulate.tabulate(result[0], headers='keys'))

    assert len(result[0]) == 40


def test_generate_labels_enhanced():
    generator = EnhancedSimilarityDatasetLabelGenerator(spectra_per_compounds=2, compound_limit=10)

    result = generator.generate_dataframe("not important")

    assert result is not None
    assert result[0] is not None
    print(tabulate.tabulate(result[0], headers='keys'))

    assert len(result[0]) == 40
