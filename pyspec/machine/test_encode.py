import multiprocessing

import pytest
from pymzml.spec import Spectrum

from pyspec.parser.pymzl.filters import MSMinLevelFilter
from pyspec.loader.binbase import BinBaseLoader
from pyspec.machine.spectra import Encoder
from pyspec.parser.pymzl.msms_finder import MSMSFinder


def test_encode():
    # 1 load some binbase spectra
    binbase = BinBaseLoader()
    data = binbase.load_spectra_for_bin_as_list(13, 1000)

    encoder = Encoder(intensity_max=10000000, min_mz=80, max_mz=500)
    encoded = encoder.encodes(data, )


sources = [
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/mzml/B1_SA0001_TEDDYLipids_Pos_1RAR7_MSMS.mzml",
    # should download this file
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch1/B1_SA0002_TEDDYLipids_Pos_1GZR9_.mzml",
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch1/B1_SA0003_TEDDYLipids_Pos_1AN1N_MSMS.mzml",
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch1/B1_SA0005_TEDDYLipids_Pos_1292V_MSMS.mzml",
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch1/B1_SA0006_TEDDYLipids_Pos_15GW2_MSMS.mzml",
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch1/B1_SA0007_TEDDYLipids_Pos_12VAU_MSMS.mzml",
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch1/B1_SA0008_TEDDYLipids_Pos_131SH_MSMS.mzml",
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch1/B1_SA0009_TEDDYLipids_Pos_24A2V_MSMS.mzml",
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch1/B1_SA0010_TEDDYLipids_Pos_24A2Y_MSMS.mzml",
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch1/B2_SA596_TEDDYLipids_Neg_14F1H_40CEmsms.mzml",
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch1/B2_SA596_TEDDYLipids_Neg_14F1H_targeted40CEMSMS.mzml",
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch1/B2_SA597_TEDDYLipids_Neg_1HXUL_40CEmsms.mzml",
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch1/B2_SA597_TEDDYLipids_Neg_1HXUL_targeted40CEMSMS.mzml",
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch1/B2_SA598_TEDDYLipids_Neg_1GFRV_40CEmsms.mzml",
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch1/B2_SA598_TEDDYLipids_Neg_1GFRV_targeted40CEMSMS.mzml",
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch1/B2_SA599_TEDDYLipids_Neg_1FZJ7_40CEmsms.mzml",
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch1/B2_SA599_TEDDYLipids_Neg_1FZJ7_targeted40CEMSMS.mzml",
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch1/B2_SA600_TEDDYLipids_Neg_1HSA9_40CEmsms.mzml",
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/teddy/batch1/B2_SA600_TEDDYLipids_Neg_1HSA9_targeted40CEMSMS.mzml",
    "http://eclipse.fiehnlab.ucdavis.edu/D%3A/lunabkup/mzml/130318_CitratePlasma_Pos_1uL_CE40_MSMS_004.mzml"

    # should download this file

]


@pytest.mark.parametrize("source", sources)
def test_encode_msms(source):
    finder = MSMSFinder()

    encoder = Encoder(intensity_max=1000, min_mz=0, max_mz=2000, directory="data/encoded")

    data = []

    def callback(msms: Spectrum, file_name: str):
        nonlocal data
        data.append(msms.convert(msms))

    finder.locate(msmsSource=source, callback=callback, filters=[MSMinLevelFilter(2)])

    from joblib import Parallel, delayed

    Parallel(n_jobs=multiprocessing.cpu_count())(delayed(encoder.encode)(x) for x in data)
