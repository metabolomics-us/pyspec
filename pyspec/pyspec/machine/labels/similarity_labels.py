import random
from typing import Optional, NamedTuple

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


class SimilarityTuple(NamedTuple):
    """
    contains different similarity measures. These need to be 0 by default or the model will break
    """
    reverse_similarity: Optional[float] = 0

    msms_spectrum_similarity: Optional[float] = 0

    precursor_distance: Optional[float] = 0

    retention_index_distance: Optional[float] = 0


class SimilarityDatasetLabelGenerator(LabelGenerator):
    """
    generates a dataset dedicated for similarity searches
    """

    def __init__(self, resample: Optional[int] = None, limit: Optional[int] = None):
        db.create_tables([MZMLSampleRecord, MZMLMSMSSpectraRecord, MZMZMSMSSpectraClassificationRecord])

        if resample is None:
            self.resampling = 1
        else:
            self.resampling = resample

        self.limit = limit

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
        names = "select distinct value as name from mzmzmsmsspectraclassificationrecord where category = 'name' order by name"

        cursor = db.connection().cursor()
        try:
            cursor.execute(names)

            row = cursor.fetchone()

            spectra: Optional[DataFrame] = None
            while row is not None:
                sample = row[0]
                if self.limit is None:
                    s = read_sql_query(
                        "select spectra_id, msms,ri,precursor,precursor_intensity,base_peak,base_peak_intensity,ion_count,value as name from mzmlmsmsspectrarecord a, mzmzmsmsspectraclassificationrecord b where a.id = b.spectra_id and b.category = 'name' and b.value = '{}'".format(
                            sample),
                        db.connection())
                else:
                    s = read_sql_query(
                        "select spectra_id, msms,ri,precursor,precursor_intensity,base_peak,base_peak_intensity,ion_count,value as name from mzmlmsmsspectrarecord a, mzmzmsmsspectraclassificationrecord b where a.id = b.spectra_id and b.category = 'name' and b.value = '{}' LIMIT {}".format(
                            sample,
                            self.limit),
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
                content_similarities.append(self.compute_similarities(row['file'][0], row['file'][1]))

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
    def compute_similarities(library: Spectra, unknown: Spectra) -> SimilarityTuple:
        """
        computes several different similaritiy scores
        :param value:
        :param random_spectra_different_compound:
        :return:
        """

        l = MSMSSpectrum(library.spectra, precursor_mz=library.precursor)
        u = MSMSSpectrum(unknown.spectra, precursor_mz=unknown.precursor)
        return SimilarityTuple(
            reverse_similarity=u.reverse_similarity(l, 1),
            msms_spectrum_similarity=u.spectral_similarity(l, 1),
            precursor_distance=unknown.precursor - library.precursor,
            retention_index_distance=unknown.ri - library.ri
        )
