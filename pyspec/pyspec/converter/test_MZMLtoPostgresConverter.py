from unittest import TestCase

import pytest

from pyspec.converter.mzml_to_postgres import MZMLtoPostgresConverter

sources = [
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch2/negative/B2b_SA1594_TEDDYLipids_Neg_MSMS_1U2WN.mzml",
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch1/positive/B1A_SA0001_TEDDYLipids_Pos_1RAR7_MSMS.mzml",
]


@pytest.mark.parametrize("source", sources)
@pytest.mark.parametrize("redo", [1, 2, 3])
def test_convert(source, redo):
    converter = MZMLtoPostgresConverter()
    converter.convert(source)
