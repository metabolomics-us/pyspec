import random
from unittest import TestCase
import numpy as np
import pytest

from pyspec.machine.labels.similarity_labels import SimilarityTuple


@pytest.mark.parametrize("top_ions", [5, 10, 12, 14, 18])
@pytest.mark.parametrize("top_ions_2", [5, 10, 9, 3, 12])
def test_to_nd(top_ions, top_ions_2):
    """

    :param self:
    :return:
    """

    test = SimilarityTuple(
        reverse_similarity=1,
        msms_spectrum_similarity=1,
        precursor_distance=1,
        retention_index_distance=1,
        top_ions=top_ions,
        library_top_ions=np.array(random.sample(range(0, 100), top_ions)),
        unknown_top_ions=np.array(random.sample(range(0, 100), top_ions_2))
    )

    assert test.top_ions == top_ions
    result = test.to_nd()

    print(result)

    assert len(result) == 7 + top_ions + top_ions - 3
    assert test.compute_size() == 7 + top_ions + top_ions - 3
