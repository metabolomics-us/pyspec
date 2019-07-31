from unittest import TestCase

from pytest import fail

from pyspec.machine.share.s3 import S3Share


def test_submit():
    share = S3Share()
    share.submit("clean_dirty")
    assert share.exists("clean_dirty")


def test_retrieve():
    share = S3Share()
    assert share.exists("clean_dirty")
    share.retrieve("clean_dirty", "test_retrieved", force=True)


def test_retrieve_no_overwrite():
    share = S3Share()

    share.submit("clean_dirty")

    assert share.exists("clean_dirty")

    share.retrieve("clean_dirty", "test_retrieved", force=True)

    try:
        share.retrieve("clean_dirty", "test_retrieved", force=False)
        fail()
    except FileExistsError as e:
        pass
