from pyspec.machine.labels.generate_labels import SimilarityDatasetLabelGenerator
from pyspec.machine.model.multi_cnn import SimilarityModel, Resnet50SimilarityModel
from pyspec.machine.spectra import Encoder, SingleEncoder


def test_predict_resnet50():
    dataset = SimilarityDatasetLabelGenerator(limit=5000)

    model = Resnet50SimilarityModel(
        width=125,
        height=125,
        channels=3,
        plots=True
    )

    model.train("similarity_test", generator=dataset, encoder=SingleEncoder(), gpus=1, epochs=50)


def test_predict():
    dataset = SimilarityDatasetLabelGenerator(limit=5000)

    model = SimilarityModel(
        width=125,
        height=125,
        channels=3,
        plots=True
    )

    model.train("similarity_test", generator=dataset, encoder=SingleEncoder(), gpus=1, epochs=50)
