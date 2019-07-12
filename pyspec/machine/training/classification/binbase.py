from typing import List

from pyspec.loader import Spectra
from pyspec.loader.binbase import BinBaseLoader
from pyspec.machine.spectra import Encoder
from pyspec.machine.training.classification.classifier import Classifier


class BinBaseBinClassifier(Classifier):
    """
    connects to the binbase database, exports all knowns bins. Each of them will be generate one folder with
    annotated spectra to them in a sub folder.
    """

    def __init__(self, name: str, encoder: Encoder, output: str, config_file="database.ini", sample_count=1000,
                 bin_ids: List = []):
        super().__init__(name, encoder, output)
        self.loader = BinBaseLoader(config_file=config_file)
        self.sample_count = int(sample_count)
        self.bin_ids = bin_ids

    def classify(self):
        connection = self.loader.connection
        # 1. query all bins with InChI Key

        if len(self.bin_ids) == 0:
            # need to load all bin ids
            cursor = connection.cursor()

            cursor.execute("select bin_id from bin where inchi_key is not null and inchi_key != '' limit 3")
            row = cursor.fetchone()

            while row is not None:
                self.bin_ids.append(int(row[0]))
                row = cursor.fetchone()
            cursor.close()

        # 2. query all samples for this bin id

        for bin_id in self.bin_ids:
            # 3. pick N random samples
            cursor = connection.cursor()
            cursor.execute(
                "select spectra.sample_id, samples.sample_name, bin.inchi_key, bin.name, spectra.spectra from spectra, samples, bin where bin.bin_id = {} and samples.visible = 'TRUE' and spectra.sample_id = samples.sample_id and spectra.bin_id = bin.bin_id  order by random() limit {}".format(
                    bin_id, self.sample_count))
            # 4. fetch these annotations with bin_id, inchi key, sample_name, spectra

            row = cursor.fetchone()

            while row is not None:
                # 5. generate folder name based on InChI Key
                sample_id, sample_name, inchi_key, name, spectra = row

                self.encoder.encode(Spectra(
                    name=name,
                    spectra=spectra
                ), prefix="{}/{}".format(inchi_key, sample_name))

                row = cursor.fetchone()
            # 6. generate sub folders, based on sample_name/spectra/splash.png
            cursor.close()
