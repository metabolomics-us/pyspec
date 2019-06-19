from typing import NamedTuple, Optional


class Spectra(NamedTuple):
    """
    basic spectra
    """
    spectra: str
    name: Optional[str] = None
    ms_level: int = 1
    inchiKey: Optional[str] = None
    splash: Optional[str] = None
    properties: dict = {}
