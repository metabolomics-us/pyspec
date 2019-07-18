from peewee import *


def connect_to_db():
    """
    connects to our database to be used for machine learning
    :return:
    """
    return SqliteDatabase('spectra.db')


db = connect_to_db()


class SpectraProperties(Model):
    """
    classification of spectra
    """

    splash: CharField()

    dirty: BooleanField()

    class Meta:
        database = db
