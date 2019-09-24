from pyspec.converter.dataset_to_postgres import DatesetToPostgresConverter


def test_convert_clean_dirty():
    converter = DatesetToPostgresConverter()

    assert converter.convert_dataset("clean_dirty") > 0
