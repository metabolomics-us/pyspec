from typing import List, NamedTuple, Optional


class Spectra(NamedTuple):
    """
    basic spectra
    """
    spectra: str
    id: Optional[str] = None
    name: Optional[str] = None
    ms_level: int = 1
    inchiKey: Optional[str] = None
    splash: Optional[str] = None
    properties: dict = {}
    submitter: Optional[dict] = None
    library: Optional[dict] = None
