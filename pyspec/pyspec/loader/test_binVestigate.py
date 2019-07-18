from pytest import fail

from pyspec.loader.binvestigate import BinVestigate


def test_load_exists():
    bv = BinVestigate()
    data = bv.load(13)

    assert data.name == "stearic acid"


def test_load_does_not_exist():
    bv = BinVestigate()

    try:
        data = bv.load(-1)
        fail()
    except Exception as e:
        pass
