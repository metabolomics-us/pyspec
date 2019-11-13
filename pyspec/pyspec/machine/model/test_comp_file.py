##
# general parameters for the model definition
import pandas
from pandas import DataFrame

from pyspec.loader import Spectra
from pyspec.machine.labels.similarity_labels import EnhancedSimilarityDatasetLabelGenerator
from pyspec.machine.model.multi_cnn import Resnet50SimilarityModel
from pyspec.machine.spectra import SingleEncoder

EPOCHS = 2  # 10
COMPOUNDS = 10 # None
SPECTRA = 2  # 50
RESAMPLE = 2  # 5
BS = 32
NO_RI = True


def test_comp_file():
    """
    tests a pre trained model against the comp file
    :return:
    """

    MODEL_NAME = "similarity_test_spectra_{}_{}_{}_{}_{}_{}".format(EPOCHS, COMPOUNDS, SPECTRA, RESAMPLE, BS, NO_RI)

    print("loading datafile for evaluation")
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

    model = Resnet50SimilarityModel(
        width=125,
        height=125,
        channels=3,
        plots=True,
        batch_size=BS
    )

    try:
        if model.is_model_trained(MODEL_NAME) is False:
            print("require to train model")
            # train model if not done so
            dataset = EnhancedSimilarityDatasetLabelGenerator(spectra_per_compounds=SPECTRA, compound_limit=COMPOUNDS,
                                                              resample=RESAMPLE, no_ri=NO_RI)

            model.train(MODEL_NAME, generator=dataset,
                        encoder=SingleEncoder(), gpus=3,
                        epochs=EPOCHS)
            m = model.get_model(MODEL_NAME)
        else:
            print("using pre trained model...")
            m = model.get_model(MODEL_NAME)
    except Exception as e:
        print("exception forces to train model")
        # train model if not done so
        dataset = EnhancedSimilarityDatasetLabelGenerator(spectra_per_compounds=SPECTRA, compound_limit=COMPOUNDS,
                                                          resample=RESAMPLE, no_ri=NO_RI)

        model.train(MODEL_NAME, generator=dataset,
                    encoder=SingleEncoder(), gpus=3,
                    epochs=EPOCHS)
        m = model.get_model(MODEL_NAME)

    print("loading model for evaluation")

    # evaluate datafile against each other
    result = []
    for unknown in spectra:
        for library in spectra:
            score = model.predict(MODEL_NAME, unknown, library, SingleEncoder(), model=m,
                                  no_ri=NO_RI)
            if unknown.name == library.name:
                print(
                    "from: {} to: {} score: {:.4f}   <= should be perfect match".format(unknown.name, library.name,
                                                                                        float(score)))
            elif score > 0.0001:
                print("from: {} to: {} score: {:.4f}".format(unknown.name, library.name, float(score)))

            result.append({
                'unknown': unknown.name,
                'library': library.name,
                'score': score
            })
        print("")

    out = DataFrame(result)
    out.to_csv("{}.csv".format(MODEL_NAME))
