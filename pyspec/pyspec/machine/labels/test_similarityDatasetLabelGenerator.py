import tabulate
import matplotlib.pyplot as plt
from skimage.color import rgba2rgb
from pyspec.machine.labels.similarity_labels import SimilarityDatasetLabelGenerator, \
    EnhancedSimilarityDatasetLabelGenerator
from pyspec.machine.spectra import SingleEncoder


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

    g = generator.get_data_generator(result[0], 128, 128, 32, SingleEncoder())

    for x in g:
        for b in range(0, 32):
            first = x[0][0][b]
            second = x[0][1][b]
            similarities = x[0][2][b]

            fig2 = plt.figure()
            ax2 = fig2.add_subplot(1, 2, 1)
            plt.imshow(first)
            ax2.set_title("first")
            ax3 = fig2.add_subplot(1, 2, 2)
            ax3.set_title("second")
            plt.imshow(second)

            plt.show()
            print("")
