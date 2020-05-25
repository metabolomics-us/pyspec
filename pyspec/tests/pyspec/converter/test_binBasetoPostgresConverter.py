from unittest import TestCase

from pyspec.converter.binbase_to_postgres import BinBasetoPostgresConverter


def test_convert():
    converter = BinBasetoPostgresConverter()
    converter.convert(pattern="050819asds%")
