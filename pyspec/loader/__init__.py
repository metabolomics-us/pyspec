from typing import NamedTuple


class Spectra(NamedTuple):
    """
    basic spectra
    """
    spectra: str
    name: str
    ms_level: int = 1
    properties: dict = {}
