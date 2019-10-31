from peewee import *
from playhouse.shortcuts import ReconnectMixin

from pyspec import config


class ReconnectPostgresDatabase(ReconnectMixin, PostgresqlDatabase):
    """
    tries to reconnect automatically for us
    """
    reconnect_errors = (
        # Error class, error message fragment (or empty string for all).
        (OperationalError, ''),  # MySQL server has gone away.
    )

    pass


def connect_to_db():
    """
    connect to our internal database
    :return: 
    """
    c = config.config(filename="database.ini", section="machine")

    return ReconnectPostgresDatabase(c.get('database'), user=c.get('user'), host=c.get('host'),
                                    password=c.get('password'))


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
        indexes = (
            (('name'), False)
        )


class MZMLMSMSSpectraRecord(Model):
    """

    """

    sample = ForeignKeyField(MZMLSampleRecord, backref='msms', on_delete='cascade')

    splash = CharField()

    msms = TextField()

    rt = DoubleField()

    precursor = DoubleField()

    precursor_intensity = DoubleField()

    precursor_charge = IntegerField()

    base_peak = DoubleField()

    base_peak_intensity = DoubleField()

    ion_count = IntegerField()

    level = IntegerField()

    scan_number = IntegerField()

    ri = DoubleField()

    class Meta:
        indexes = (
            (('splash'), False)
        )
        database = db


class MZMZMSMSSpectraClassificationRecord(Model):
    """
    classification record if a spectra is clean or dirty
    """

    # linked spectra
    spectra = ForeignKeyField(MZMLMSMSSpectraRecord, backref='classification', on_delete='cascade')

    # is this a predicted value
    predicted = BooleanField(default=False)

    # category of the classification
    category = CharField()

    # value of the classification
    value = CharField()

    class Meta:
        database = db
        indexes = (
            (('category'), False)
        )
