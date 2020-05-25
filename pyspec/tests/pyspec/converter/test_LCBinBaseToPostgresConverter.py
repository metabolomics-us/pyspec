from unittest import TestCase

from pyspec.converter.carrot_msms_dump_to_postgres import LCBinBaseToPostgresConverter


def test_convert_dump():
    converter = LCBinBaseToPostgresConverter()
    count = converter.convert("../datasets/B2b_SA1594_TEDDYLipids_Neg_MSMS_1U2WN.msms.json.gz", compressed=True)

    assert count > 0
