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
    ri: Optional[float] = None
    intensity: Optional[float] = None
    ionCount: Optional[float] = None
    precursorIntensity: Optional[float] = None
    precursor: Optional[float] = None
    basePeak: Optional[float] = None
    basePeakIntensity: Optional[float] = None
