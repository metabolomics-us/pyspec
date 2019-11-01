import traceback

import pytest

from pyspec.machine.labels.similarity_labels import EnhancedSimilarityDatasetLabelGenerator
from pyspec.machine.model.multi_cnn import Resnet50SimilarityModel
from pyspec.machine.spectra import SingleEncoder


@pytest.mark.parametrize("limit", [3, 5, 10])
@pytest.mark.parametrize("resample", [1, 3, 5])
@pytest.mark.parametrize("batchsize", [16, 32, 64])
def test_predict_resnet50(limit, resample, batchsize):
    dataset = EnhancedSimilarityDatasetLabelGenerator(compound_limit=100, spectra_per_compounds=limit,
                                                      resample=resample)

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

    data = dataset.generate_dataframe("similarity_test_spectra:{}_resample:{}".format(limit, resample))[0]

    m = model.get_model("similarity_test_spectra:{}_resample:{}".format(limit, resample))

    def predictor(x):
        try:
            result = model.predict("similarity_test_spectra:{}_resample:{}".format(limit, resample),
                                   encode=SingleEncoder(),
                                   first=x['file'][0], second=x['file'][1], model=m)

            print(result)
        except Exception as e:
            traceback.print_stack()
            raise e

    data.tail(10).apply(predictor, axis=1)
