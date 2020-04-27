import multiprocessing

import pytest
from pymzml.spec import Spectrum

from pyspec.machine.spectra import DualEncoder
from pyspec.parser.pymzl.filters import MSMinLevelFilter
from pyspec.parser.pymzl.msms_finder import MSMSFinder

sources = [
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch1/positive/B1A_SA0001_TEDDYLipids_Pos_1RAR7_MSMS.mzml"
]


@pytest.mark.parametrize("source", sources)
def test_encode_msms(source):
    finder = MSMSFinder()

    encoder = DualEncoder(intensity_max=1000, min_mz=0, max_mz=2000, directory="data/encoded")

    data = []

    def callback(msms: Spectrum, file_name: str):
        nonlocal data
        if msms is not None:
            data.append(msms.convert(msms))

    finder.locate(msmsSource=source, callback=callback, filters=[MSMinLevelFilter(2)])

    from joblib import Parallel, delayed

    Parallel(n_jobs=multiprocessing.cpu_count())(delayed(encoder.encode)(x) for x in data)
