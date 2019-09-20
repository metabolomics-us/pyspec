from peewee import *

from pyspec import config


def connect_to_db():
    """
    connect to our internal database
    :return: 
    """
    c = config.config(filename="database.ini", section="machine")

    return PostgresqlDatabase(c.get('database'), user=c.get('user'), host=c.get('host'), password=c.get('password'))


db = connect_to_db()


class MZMLSampleRecord(Model):
    """

    """

    # complete path
    file_name = CharField(unique=True)

    # shortened name
    name = CharField()
    
    instrument = CharField()

    class Meta:
        database = db


class MZMLMSMSSpectraRecord(Model):
    """

    """

    sample = ForeignKeyField(MZMLSampleRecord, backref='msms', on_delete='cascade')

    splash = CharField()

    msms = TextField()

    rt = DoubleField()

    base_peak = DoubleField()

    base_peak_intensity = DoubleField()

    ion_count = IntegerField()

    level = IntegerField()

    class Meta:
        database = db


class MZMLMSMSSpectraCleanDirtyClassificationRecord(Model):
    """
    classification record if a spectra is clean or dirty
    """

    # linked spectra
    spectra = ForeignKeyField(MZMLMSMSSpectraRecord, backref='classification_dirty', on_delete='cascade')

    # is this a predicted value
    predicted = BooleanField(default=False)

    # if it is a predicted value, which model was used
    model = CharField()

    # if it is a predicted value, which encoder was used
    encoder = CharField()

    # is this spectra clean or dirty
    dirty = BooleanField()

    class Meta:
        database = db
