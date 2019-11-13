from unittest import TestCase
import numpy as np

from pyspec.machine.labels.similarity_labels import SimilarityTuple


def test_to_nd():
    """

    :param self:
    :return:
    """
    test = SimilarityTuple(
        reverse_similarity=1,
        msms_spectrum_similarity=1,
        precursor_distance=1,
        retention_index_distance=1,
        top_ions=5,
        library_top_ions=np.array([1, 2, 3, 4, 5]),
        unknown_top_ions=np.array([1, 2, 3, 4, 5])
    )

    assert test.top_ions == 5
    result = test.to_nd()

    print(result)

    assert len(result) == 14
    assert test.compute_size() == 14