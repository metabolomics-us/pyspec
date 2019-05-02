from unittest import TestCase

from pytest import fail

from pyspec.parser.parse_bv_report import ParseReport


def test_parse_file_fail_no_file():
    "tests parsing of the given file"

    report = ParseReport()
    try:
        report.parse_file("i_do_not_exist.txt")
        fail()
    except Exception:
        pass


def test_parse_file_success():
    report = ParseReport()

    data = report.parse_file("test.txt")

    assert data
    assert len(data) == 10

    assert data[0].name == "alanine"
    assert data[0].spectra != ""
