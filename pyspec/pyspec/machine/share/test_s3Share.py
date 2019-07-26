from unittest import TestCase

from pyspec.machine.share.s3 import S3Share


def test_submit_retrieve():
    share = S3Share()

    share.submit("clean_dirty")

    assert share.exists("clean_dirty")