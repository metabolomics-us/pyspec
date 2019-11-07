##
# general parameters for the model definition
import pandas
from pandas import DataFrame

from pyspec.loader import Spectra
from pyspec.machine.labels.similarity_labels import EnhancedSimilarityDatasetLabelGenerator
from pyspec.machine.model.multi_cnn import Resnet50SimilarityModel
from pyspec.machine.spectra import SingleEncoder

EPOCHS = 1
COMPOUNDS = 100
SPECTRA = 3
RESAMPLE = 10
BS = 32
NO_RI = True


def test_comp_file():
    """
    tests a pre trained model against the comp file
    :return:
    """

    MODEL_NAME = "similarity_test_spectra_{}_{}_{}_{}_{}_{}".format(EPOCHS, COMPOUNDS, SPECTRA, RESAMPLE, BS, NO_RI)

    # datafile loading
    dataframe: DataFrame = pandas.read_csv("datasets/machine_komp_test_data.csv")

    spectra = []

    for index, row in dataframe.iterrows():
        spectra.append(Spectra(
            name=row['name'],
            spectra=row['msms'],
            ms_level=2,
            precursor=row['mz']

        ))
    # train model if not done so
    dataset = EnhancedSimilarityDatasetLabelGenerator(spectra_per_compounds=SPECTRA, compound_limit=COMPOUNDS,
                                                      resample=RESAMPLE, no_ri=NO_RI)

    model = Resnet50SimilarityModel(
        width=125,
        height=125,
        channels=3,
        plots=True,
        batch_size=BS
    )

    model.train(MODEL_NAME, generator=dataset,
                encoder=SingleEncoder(), gpus=3,
                epochs=EPOCHS)

    m = model.get_model(MODEL_NAME)

    # evaluate datafile against each other
    for unknown in spectra:
        for library in spectra:
            score = model.predict(MODEL_NAME, unknown, library, SingleEncoder(), model=m,
                                  no_ri=NO_RI)
            print("from: {} to: {} score: {}".format(unknown.name, library.name, score))
