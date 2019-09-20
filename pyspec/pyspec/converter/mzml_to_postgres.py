from pymzml.spec import Spectrum as PySpectrum
from splash import SpectrumType, Splash, Spectrum

from pyspec.machine.persistence.model import MZMLSampleRecord, MZMLMSMSSpectraRecord, db
from pyspec.parser.pymzl.filters import MSMinLevelFilter
from pyspec.parser.pymzl.msms_finder import MSMSFinder


class MZMLtoPostgresConverter:
    """
    converts mzml data file and stores them in a postgres database
    for later analysis
    """

    def __init__(self):
        db.create_tables([MZMLSampleRecord, MZMLMSMSSpectraRecord])

    def convert(self, input: str):
        """
        converts the given input and stores it at the defined postgres database location
        :param input:
        :return:
        """

        finder = MSMSFinder()

        def callback(msms: PySpectrum, file_name: str):
            with db.atomic() as transaction:
                if msms is None:
                    # 1. check if sample exist, if yes delete it
                    try:
                        record = MZMLSampleRecord.get(MZMLSampleRecord.file_name == file_name)
                        record.delete_instance()
                    except Exception:
                        # object doesn't exist
                        pass
                    # 2. create sample object
                    MZMLSampleRecord.create(file_name=file_name, instrument="")

                else:
                    # 3. load sample object
                    record = MZMLSampleRecord.get(MZMLSampleRecord.file_name == file_name)
                    # 3. associated msms spectra to it

                    # 4. commit transaction
                    highest = msms.highest_peaks(1)[0]
                    spectra = msms.convert(msms).spectra

                    splash = Splash().splash(Spectrum(spectra, SpectrumType.MS))

                    spectra = MZMLMSMSSpectraRecord.create(sample=record, msms=spectra, rt=msms.scan_time[0],
                                                           splash=splash,
                                                           level=msms.ms_level, base_peak=highest[0],
                                                           base_peak_intensity=highest[1],
                                                           ion_count=len(msms.peaks("centroided")))

        finder.locate(msmsSource=input, callback=callback, filters=[MSMinLevelFilter(2)])
