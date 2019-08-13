from pytest import fail

from pyspec.parser.bv.parse_bv_report import ParseReport


def test_parse_file_fail_no_file():
    "tests parsing of the given file"

    report = ParseReport()
    try:
        report.parse_file("i_do_not_exist.txt")
        fail()
    except Exception:
        pass
