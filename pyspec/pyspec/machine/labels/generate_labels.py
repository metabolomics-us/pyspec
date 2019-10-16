import csv
import os
import random
from abc import abstractmethod
from glob import iglob
from typing import Tuple, Optional, List

from keras.utils import Sequence
from keras_preprocessing.image import ImageDataGenerator
from pandas import DataFrame, read_sql_query

from pyspec.loader import Spectra
from pyspec.machine.labels.spectra_generator import SpectraDataGenerator
from pyspec.machine.persistence.model import db, MZMLSampleRecord, MZMLMSMSSpectraRecord, \
    MZMZMSMSSpectraClassificationRecord
from pyspec.machine.spectra import Encoder


class LabelGenerator:
    """
    class to easily generate a file for us containing all the labels
    this is based on pictures in directories


    """

    @abstractmethod
    def generate_labels(self, input: str, callback, training: bool):
        """
        :param input: the input file to utilize
        :param callback: def callback(identifier, class)
        :return:
        """

    def contains_test_data(self) -> bool:
        """
        does the label generator provide it's own test data
        :return:
        """
        return True

    def generate_dataframe(self, input: str) -> Tuple[DataFrame, Optional[DataFrame]]:
        """
        generates a dataframe for the given input with all the internal labels. This will be used for training and validation
        :param input:
        :return:
        """
        data = []

        def callback(id, category, training: bool):
            nonlocal data

            if self.is_file_based():
                assert os.path.exists(id), 'please ensure all files exist. Missing {}'.format(id)

            data.append({
                "file": id,
                "class": category,
                "training": training
            })

        self.generate_labels(input, callback, training=True)

        training = DataFrame(list(filter(lambda x: x['training'] is True, data)))

        if self.contains_test_data():
            self.generate_labels(input, callback, training=False)
            testing = DataFrame(list(filter(lambda x: x['training'] is False, data)))
        else:
            testing = None

        return training, testing

    def returns_multiple(self):
        "do we return multiple fields for our 'file' column and the model needs to flatten it"
        return False

    def is_file_based(self) -> bool:
        """
        if this generator is based on files
        :return:
        """
        return True

    def to_csv(self, input: str, file_name: str, training: bool):
        """
        reads all the images, and saves them as a CSV file
        :param input: from where to load the data
        :param file_name: name of the labeled datafile
        :return:
        """
        result = self.generate_dataframe(input)

        if training is True:
            result[0].to_csv(file_name, encoding='utf-8', index=False)
        else:
            result[1].to_csv(file_name, encoding='utf-8', index=False)

    def get_data_generator(self, dataframe: DataFrame, width: int, height: int, batch_size: int, encoder: Encoder,
                           class_mode: str = 'categorical'):
        """

        returns the correct data generator for this generated training data set

        :param dataframe:
        :param width:
        :param height:
        :param batch_size:
        :param class_mode:
        :return:
        """
        datagen = ImageDataGenerator()
        return datagen.flow_from_dataframe(
            dataframe=dataframe,
            directory=None,
            x_col='file',
            y_col='class',
            target_size=(width, height),
            class_mode=class_mode,
            batch_size=batch_size
        )


class DirectoryLabelGenerator(LabelGenerator):
    """

    generates labels from pictures in a directory
    , which needs to be configured like this

    dataset_name/train/class
    dataset_name/test/class

    for example

    dataset_spectra/train/clean
    dataset_spectra/train/dirty
    dataset_spectra/test/clean
    dataset_spectra/test/dirty

    """

    def generate_labels(self, input: str, callback, training: bool):

        data = "{}/train".format(input) if training else "{}/test".format(input)

        for category in os.listdir(data):
            for file in iglob("{}/{}/**/*.png".format(data, category), recursive=True):
                callback(file, category, training)


class CSVLabelGenerator(LabelGenerator):
    """
    generates labels from a CSV file
    """

    def generate_labels(self, input: str, callback, training: bool):
        import os
        assert os.path.exists(input), "please ensure that {} exists!".format(input)
        input_file = os.path.join(input, "train.csv") if training else os.path.join(input, "test.csv")
        assert os.path.isfile(input_file), "please ensure that {} is a file!".format(input_file)

        print("using: {}".format(input_file))
        with open(input_file, mode='r') as infile:
            reader = csv.reader(infile)

            # first row is headers

            row = next(reader)

            assert len(row) >= 2, "please ensure you have more than 2 columns!, But given where {}, '{}'".format(
                len(row), row)

            if row[0] == self.field_category:
                c = 0
                f = 1
            elif row[1] == self.field_category:
                c = 1
                f = 0
            else:
                assert False, "please ensure that your column names are {} and {} instead of {}".format(
                    self.field_category, self.field_id, row)

            for row in reader:
                if os.path.exists(row[f]):
                    file = row[f]
                elif os.path.exists("{}/{}".format(input, row[f])):
                    file = "{}/{}".format(input, row[f])
                else:
                    raise Exception("sorry we did not find the file: {} or {}/{}".format(row[f], input, row[f]))

                callback(file, row[c], training)

    def __init__(self, field_id: str = "file", field_category: str = "class"):
        self.field_id = field_id
        self.field_category = field_category


class MachineDBDataSetGenerator(LabelGenerator):
    """
    generates a dataset based on the PeeWee domain classes. Query is done direcly as SQL to reduce overhead
    """

    def __init__(self, fields=['msms'], query: Optional[str] = None):
        """

        :param fields: which fields you want to select and map. If more than one, the result fieled in the dataframe, will be a list!
        :param query: your query to execute, it has to return a column name 'class' and take one input argument
        """

        db.create_tables([MZMLSampleRecord, MZMLMSMSSpectraRecord, MZMZMSMSSpectraClassificationRecord])
        self.fields = fields

        if query is None:
            self.query = "select a.*, b.value as class from  mzmlmsmsspectrarecord a, mzmzmsmsspectraclassificationrecord b where a.id = b.spectra_id and category = '{}'"
        else:
            assert "class" in query, "please ensure that a class field is in the query"
            self.query = query

    def get_fields(self) -> List[str]:
        """
        the fields which are returned under the 'id' as list, if more than 1
        :return:
        """
        return self.fields

    def generate_labels(self, input: str, callback, training: bool):
        """
        input is the name of dataset, example 'clean_dirty' which translates to the column 'category' in the database
        :param input:
        :param callback:
        :param training:
        :return:
        """
        input = input.split("/")[-1]

        result = read_sql_query(self.query
            .format(
            input),
            db.connection())

        for index, row in result.iterrows():
            data = []

            for y in self.fields:
                data.append(row[y])

            if len(data) > 1:
                callback(
                    id=data,
                    category=row['class'],
                    training=training
                )
            else:
                callback(
                    id=data[0],
                    category=row['class'],
                    training=training
                )

    def returns_multiple(self):
        """
        yes we can return multiple data inputs
        :return:
        """
        return len(self.fields) > 1

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

        # contains all spectra and compounds
        if self.limit is None:
            spectra = read_sql_query(
                "select spectra_id, msms,ri,precursor,precursor_intensity,base_peak,base_peak_intensity,ion_count,value as name from mzmlmsmsspectrarecord a, mzmzmsmsspectraclassificationrecord b where a.id = b.spectra_id and b.category = 'name'",
                db.connection())
        else:
            spectra = read_sql_query(
                "select spectra_id, msms,ri,precursor,precursor_intensity,base_peak,base_peak_intensity,ion_count,value as name from mzmlmsmsspectrarecord a, mzmzmsmsspectraclassificationrecord b where a.id = b.spectra_id and b.category = 'name' LIMIT {}".format(
                    self.limit),
                db.connection())

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
                    random_spectra_different_compound = groups.get_group(group_different_compound).sample(1).iloc[0].to_dict()

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
