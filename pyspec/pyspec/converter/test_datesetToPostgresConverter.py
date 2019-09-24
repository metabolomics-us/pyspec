from unittest import TestCase

from pyspec.converter.dataset_to_postgres import DatesetToPostgresConverter


def test_convert_clean_dirty():
    converter = DatesetToPostgresConverter()

    converter.convert_clean_dirty("clean_dirty_full")
