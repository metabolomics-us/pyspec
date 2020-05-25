from pyspec.converter.dataset_to_postgres import DatesetToPostgresConverter


def test_convert_pst_neg():
    converter = DatesetToPostgresConverter()

    converted = converter.convert_dataset("pos_neg")
    assert converted == 2543

def test_convert_clean_dirty():
    converter = DatesetToPostgresConverter()

    converted = converter.convert_dataset("clean_dirty")
    assert converted == 88
