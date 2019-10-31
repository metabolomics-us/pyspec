from pyspec.machine.labels.similarity_labels import SimilarityDatasetLabelGenerator
from pyspec.machine.labels.spectra_generator import SpectraDataGenerator
from pyspec.machine.spectra import SingleEncoder


def test_dataset_generator():
    """
    tests our generator
    :return:
    """

    dataset = SimilarityDatasetLabelGenerator(spectra_per_compounds=500)

    data = dataset.generate_dataframe("dasdas")[0]

    content = []

    def collector(row):
        nonlocal content
        content.append((row['file'][0], row['class']))

    data.apply(collector, axis=1)

    generator = SpectraDataGenerator(spectra=content, encoder=SingleEncoder())
    result = generator[0]
    result = generator[1]
    result = generator[2]
    result = generator[3]
    result
