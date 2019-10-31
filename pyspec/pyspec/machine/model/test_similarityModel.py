import pytest

from pyspec.machine.labels.similarity_labels import EnhancedSimilarityDatasetLabelGenerator
from pyspec.machine.model.multi_cnn import Resnet50SimilarityModel
from pyspec.machine.spectra import SingleEncoder


@pytest.mark.parametrize("limit", [3, 5, 10])
@pytest.mark.parametrize("resample", [1, 3, 5])
@pytest.mark.parametrize("batchsize", [16, 32, 64])
def test_predict_resnet50(limit, resample, batchsize):
    dataset = EnhancedSimilarityDatasetLabelGenerator(limit=limit, resample=resample)

    model = Resnet50SimilarityModel(
        width=125,
        height=125,
        channels=3,
        plots=True,
        batch_size=batchsize
    )

    model.train("similarity_test_spectra:{}_resample:{}".format(limit, resample), generator=dataset,
                encoder=SingleEncoder(), gpus=3,
                epochs=10)
