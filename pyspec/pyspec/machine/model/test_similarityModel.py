from pyspec.machine.labels.similarity_labels import SimilarityDatasetLabelGenerator, \
    EnhancedSimilarityDatasetLabelGenerator
from pyspec.machine.model.multi_cnn import SimilarityModel, Resnet50SimilarityModel
from pyspec.machine.spectra import Encoder, SingleEncoder


def test_predict_resnet50():
    dataset = EnhancedSimilarityDatasetLabelGenerator(limit=3500)

    model = Resnet50SimilarityModel(
        width=125,
        height=125,
        channels=3,
        plots=True
    )

    model.train("similarity_test", generator=dataset, encoder=SingleEncoder(), gpus=1, epochs=10)
