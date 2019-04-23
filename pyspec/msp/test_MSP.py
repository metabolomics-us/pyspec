from pyspec.loader import Spectra
from pyspec.msp.writer import MSP


def test_from_bin():
    msp = MSP()

    result = msp.from_bin(13)

    assert result is not None


def test_from_spectra():
    msp = MSP()
    result = msp.from_spectra(Spectra(spectra="123:134", name="test", ms_level=1))
    assert result is not None


def test_from_str():
    msp = MSP(

    )

    result = msp.from_str("123:213 1233:333", "test", {})
    assert result is not None
