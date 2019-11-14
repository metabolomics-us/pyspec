import random
import traceback
from typing import Optional, NamedTuple, List

import tabulate
from keras.utils import Sequence
from pandas import read_sql_query, DataFrame

from pyspec.loader import Spectra
from pyspec.machine.labels.generate_labels import LabelGenerator
from pyspec.machine.labels.spectra_generator import SpectraDataGenerator, SimilarityMeasureGenerator
from pyspec.machine.persistence.model import db, MZMLSampleRecord, MZMLMSMSSpectraRecord, \
    MZMZMSMSSpectraClassificationRecord
from pyspec.machine.spectra import Encoder
from pyspec.parser.pymzl.msms_spectrum import MSMSSpectrum
import numpy as np


def top_n_ions(spectra: str, top_n: int):
    """
    computes top n ions
    :param spectra:
    :param top_n:
    :return:
    """
    pairs = sorted(map(lambda x: x.split(":")[0], spectra.split(" ")), key=lambda x: x[1])
    result = np.array(pairs[0:top_n])

    return result


def padarray(A, size):
    t = size - len(A)
    if t < 0:
        # actually need to drop them
        return A[0:A.size - abs(t)]
    else:
        return np.pad(A, pad_width=(0, t), mode='constant')


class SimilarityTuple(NamedTuple):
    """
    contains different similarity measures. These need to be 0 by default or the model will break
    """
    reverse_similarity: Optional[float] = 0

    msms_spectrum_similarity: Optional[float] = 0

    precursor_distance: Optional[float] = 0

    retention_index_distance: Optional[float] = 0

    top_ions: int = 10

    library_top_ions: np.ndarray = None

    unknown_top_ions: np.ndarray = None

    def to_nd(self):
        """
        generates a similarity tuple array representation for us
        :return: 
        """
        if self.library_top_ions is None:
            self.library_top_ions = np.empty((self.top_ions, np.float))
        if self.unknown_top_ions is None:
            self.unknown_top_ions = np.empty((self.top_ions, np.float))

        library_top_ions = padarray(self.library_top_ions, self.top_ions)
        unknown_top_ions = padarray(self.unknown_top_ions, self.top_ions)

        return np.append(
            np.array([self.reverse_similarity, self.msms_spectrum_similarity, self.precursor_distance,
                      self.retention_index_distance]), np.append(
                library_top_ions,
                unknown_top_ions
            )
        )

    def compute_size(self) -> int:
        """
        computes the size of the tuple for ease of usage. Depends obviously on the utilized attribues and needs to be adjusted
        if this gets changed.
        :return:
        """
        return len(list(self)) - 3 + 2 * self.top_ions


class SimilarityDatasetLabelGenerator(LabelGenerator):
    """
    generates a dataset dedicated for similarity searches
    """

    def __init__(self, resample: Optional[int] = None, spectra_per_compounds: Optional[int] = None,
                 compound_limit: Optional[int] = None):
        db.create_tables([MZMLSampleRecord, MZMLMSMSSpectraRecord, MZMZMSMSSpectraClassificationRecord])

        if resample is None:
            self.resampling = 1
        else:
            self.resampling = resample

        assert spectra_per_compounds > 1, "please ensure that you have at least 2 spectra for each compound or the generation will fail"
        self.spectra_per_compounds = spectra_per_compounds
        self.compount_limit = compound_limit

    def generate_labels(self, input: str, callback, training: bool):
        """
        generates a label files containing the following layout

        Tuple(msms, precursor,ri,ion_count, basepeak, basepeak intensity,pre cursor, pre cursor intensity,name), Tuple(msms, precursor,ri,ion_count, basepeak, basepeak intensity,pre cursor, pre cursor intensity,name), class

        These data can than be used directly in a multi input model or further features computed
        :param input:
        :param callback:
        :param training:
        :return:
        """

        # 1. select all compounds
        if self.compount_limit is None:
            names = "select distinct value as name from mzmzmsmsspectraclassificationrecord where category = 'name' order by name"
        else:
            names = "select distinct value as name from mzmzmsmsspectraclassificationrecord where category = 'name' order by name LIMIT {}".format(
                self.compount_limit)
        cursor = db.execute_sql(names)
        try:
            row = cursor.fetchone()

            spectra: Optional[DataFrame] = None
            while row is not None:
                sample = row[0]
                if self.spectra_per_compounds is None:
                    s = read_sql_query(
                        "select spectra_id, msms,ri,precursor,precursor_intensity,base_peak,base_peak_intensity,ion_count,value as name from mzmlmsmsspectrarecord a, mzmzmsmsspectraclassificationrecord b where a.id = b.spectra_id and b.category = 'name' and b.value = '{}'".format(
                            sample),
                        db.connection())
                else:
                    s = read_sql_query(
                        "select spectra_id, msms,ri,precursor,precursor_intensity,base_peak,base_peak_intensity,ion_count,value as name from mzmlmsmsspectrarecord a, mzmzmsmsspectraclassificationrecord b where a.id = b.spectra_id and b.category = 'name' and b.value = '{}' LIMIT {}".format(
                            sample,
                            self.spectra_per_compounds),
                        db.connection())

                if spectra is None:
                    spectra = s
                else:
                    spectra = spectra.append(s)
                row = cursor.fetchone()

        finally:
            cursor.close()
        assert spectra is not None

        print("evaluating {} spectra for this label generation".format(len(spectra)))

        def function(row):
            """
            finds the related data and calls the callback. For each observed row, we add 2 rows to the main dataset
            :param row:
            :return:
            """
            try:
                value = self._convert(row.to_dict())

                for x in range(0, self.resampling):
                    group_same_compound = groups.get_group(row['name'])
                    random_spectra_same_compound = group_same_compound[
                        group_same_compound['spectra_id'] != row['spectra_id']].sample(1).iloc[0].to_dict()

                    group_different_compound = random.choice([g for g in groups.groups.keys() if g != row['name']])
                    random_spectra_different_compound = groups.get_group(group_different_compound).sample(1).iloc[
                        0].to_dict()

                    callback(
                        id=(value, self._convert(random_spectra_same_compound)),
                        category=True,
                        training=training
                    )

                    callback(
                        id=(value, self._convert(random_spectra_different_compound)),
                        category=False,
                        training=training
                    )
            except ValueError as e:
                traceback.print_exc()
                pass

        groups = spectra.groupby(['name'])
        spectra.apply(function, axis=1)

        # drop duplicated values
        spectra.drop_duplicates(inplace=True)

        pass

    def _convert(self, row: dict) -> Spectra:
        return Spectra(
            spectra=row['msms'],
            name=row['name'],
            ms_level=2,
            ri=row['ri'],
            ionCount=row['ion_count'],
            precursor=row['precursor'],
            precursorIntensity=row['precursor_intensity'],
            basePeak=row['base_peak'],
            basePeakIntensity=row['base_peak_intensity']

        )

    def returns_multiple(self):
        """
        yes we can return multiple data inputs
        :return:
        """
        return True

    def is_file_based(self) -> bool:
        """
        nope not based on files
        :return:
        """
        return False

    def contains_test_data(self) -> bool:
        """
        we don't have predefined test data, need to generate them your self using splits
        :return:
        """
        return False

    def get_data_generator(self, dataframe: DataFrame, width: int, height: int, batch_size: int, encoder: Encoder,
                           class_mode: str = 'categorical'):
        """
        generates a custom data generator
        :param dataframe:
        :param width:
        :param height:
        :param batch_size:
        :param class_mode:
        :return:
        """

        content_first = []
        content_second = []

        def collector(row):
            nonlocal content_first
            nonlocal content_second

            # first spectra object
            content_first.append((row['file'][0], row['class']))

            # second spectra object
            content_second.append((row['file'][1], row['class']))

        dataframe.apply(collector, axis=1)

        generator_first = SpectraDataGenerator(spectra=content_first, encoder=encoder)
        generator_second = SpectraDataGenerator(spectra=content_second, encoder=encoder)

        assert len(generator_first) == len(generator_second), "both generators need to be the same size!"

        class CombinedGenerator(Sequence):
            """
            combines our generators to easily generate the required input data for our model
            with multiple inputs
            """

            def __getitem__(self, index):
                X1_batch, Y_batch = generator_first.__getitem__(index)
                X2_batch, Y_batch = generator_second.__getitem__(index)

                X_batch = [X1_batch, X2_batch]

                return X_batch, Y_batch

            def __len__(self):
                return generator_first.__len__()

        return CombinedGenerator()


class EnhancedSimilarityDatasetLabelGenerator(SimilarityDatasetLabelGenerator):
    """
    generates a dataset dedicated for similarity searches, which also includes several different similarity matrix scores
    and so allows for a more complex model
    """

    def __init__(self, resample: Optional[int] = None, spectra_per_compounds: Optional[int] = None,
                 compound_limit: Optional[int] = None, no_ri: bool = False):
        super().__init__(resample=resample, spectra_per_compounds=spectra_per_compounds, compound_limit=compound_limit)
        self.no_ri = no_ri

    def get_data_generator(self, dataframe: DataFrame, width: int, height: int, batch_size: int, encoder: Encoder,
                           class_mode: str = 'categorical'):
        """
        generates a custom data generator
        :param dataframe:
        :param width:
        :param height:
        :param batch_size:
        :param class_mode:
        :return:
        """

        encoder.width = width
        encoder.height = height

        content_first = []
        content_second = []
        content_similarities = []

        def collector(row):
            nonlocal content_first
            nonlocal content_second

            if row['file'][0].precursor is not None and row['file'][1].precursor is not None and row['file'][
                0].ri is not None and row['file'][1].ri is not None:
                # first spectra object
                content_first.append((row['file'][0], row['class']))

                # second spectra object
                content_second.append((row['file'][1], row['class']))

                # compute similarity scores here to be appended
                content_similarities.append(self.compute_similarities(row['file'][0], row['file'][1], self.no_ri))

        dataframe.apply(collector, axis=1)

        generator_first = SpectraDataGenerator(spectra=content_first, encoder=encoder, batch_size=batch_size)
        generator_second = SpectraDataGenerator(spectra=content_second, encoder=encoder, batch_size=batch_size)
        generator_third = SimilarityMeasureGenerator(data=content_similarities, batch_size=batch_size, )

        assert len(generator_first) == len(generator_second), "both generators need to be the same size!"

        class CombinedGenerator(Sequence):
            """
            combines our generators to easily generate the required input data for our model
            with multiple inputs
            """

            def __getitem__(self, index):
                X1_batch, Y_batch = generator_first.__getitem__(index)
                X2_batch, Y_batch = generator_second.__getitem__(index)
                X3_batch, Y_batch = generator_third.__getitem__(index)

                X_batch = [X1_batch, X2_batch, X3_batch]

                return X_batch, Y_batch

            def __len__(self):
                return generator_first.__len__()

        return CombinedGenerator()

    @staticmethod
    def compute_similarities(library: Spectra, unknown: Spectra, no_ri: bool = False) -> SimilarityTuple:
        """
        computes several different similaritiy scores
        :param value:
        :param random_spectra_different_compound:
        :return:
        """

        l = MSMSSpectrum(library.spectra, precursor_mz=library.precursor)
        u = MSMSSpectrum(unknown.spectra, precursor_mz=unknown.precursor)
        reverse_similarity = u.reverse_similarity(l, 1)
        msms_spectrum_similarity = u.spectral_similarity(l, 1)
        precursor_distance = unknown.precursor - library.precursor

        if no_ri is False:
            retention_index_distance = unknown.ri - library.ri
        else:
            retention_index_distance = 0

        return SimilarityTuple(
            reverse_similarity=reverse_similarity,
            msms_spectrum_similarity=msms_spectrum_similarity,
            precursor_distance=precursor_distance,
            retention_index_distance=retention_index_distance,
            library_top_ions=top_n_ions(library.spectra, 10),
            unknown_top_ions=top_n_ions(unknown.spectra, 10),
            top_ions=10
        )
