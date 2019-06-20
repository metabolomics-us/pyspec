"""pytest for MS-Dial MSMS feature scraper

Tests if excel file being searched is in correct format for script.


Author: Bryan Roberts
"""

import msdial_msms
import pytest

def test_msdial_msms():
    file = msdial_msms.getExcelSheets()
    wb = msdial_msms.openWorkBook(file, 0)
    sheet = msdial_msms.makeSheet(wb)
    msdial_msms.test_msmsColumn(sheet)
    msdial_msms.test_mzColumn(sheet)
    msdial_msms.test_retentionTimeColumn(sheet)

