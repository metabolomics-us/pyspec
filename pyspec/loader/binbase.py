from typing import List

import psycopg2

from pyspec import config
from pyspec.loader import Spectra


class BinBaseLoader:
    """
    load data from binbase
    """

    def __init__(self, config_file="database.ini"):
        """
        connect to the database
        :param server:
        :param port:
        :param database:
        :param user:
        :param password:
        """
        db = config.config(filename=config_file, section="postgres")

        self.connection = psycopg2.connect(**db)

    def __del__(self):
        if self.connection is not None:
            self.connection.close()

    """
    loads data from binbase
    """

    def load_spectra_for_bin(self, bin_id, limit=100, work_on_data=None):
        """
        loads the last [limit] spectra for the given bin id from the binbase. The provided callback will do something on the data
        :param bin_id: id of bin
        :param limit: how many spectra you want
        :return:
        """
        cursor = self.connection.cursor()

        cursor.execute(
            "select a.spectra, a.retention_index, b.name from spectra a, bin b where a.bin_id= b.bin_id and a.bin_id = {} order by a.spectra_id desc limit {}".format(
                bin_id, limit))

        print("loading first....")
        row = cursor.fetchone()

        while row is not None:
            if work_on_data is not None:
                spectra = Spectra(
                    spectra=row[0],
                    name=row[2],
                    ms_level=1,
                )
                work_on_data(spectra)
            row = cursor.fetchone()

    def load_spectra_for_bin_as_list(self, bin_id, limit) -> List[Spectra]:
        """
        loads the data in a list. Please be aware that this is no memory efficient
        :param bin_id:
        :param limit:
        :return:
        """
        data = []
        self.load_spectra_for_bin(bin_id, limit, lambda x: data.append(x))
        return data
