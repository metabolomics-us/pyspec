from pyspec.machine.labels.generate_labels import SimilarityDatasetLabelGenerator
from pyspec.machine.model.multi_cnn import SimilarityModel
from pyspec.machine.spectra import Encoder


def test_predict():
    dataset = SimilarityDatasetLabelGenerator(limit=500)

    model = SimilarityModel(
        width=125,
        height=125,
        channels=3,
        plots=False
    )

    model.train("similarity_test", generator=dataset, encoder=Encoder())
