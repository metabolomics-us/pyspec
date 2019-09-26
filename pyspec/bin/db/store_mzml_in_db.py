##
# simple bin script to encode all files in a directory into images
import multiprocessing
from glob import glob1, glob, iglob
from xml.etree.ElementTree import ParseError

from pathlib import Path

import argparse
import os
from pymzml.spec import Spectrum

from pyspec.converter.mzml_to_postgres import MZMLtoPostgresConverter
from pyspec.machine.spectra import Encoder
from pyspec.parser.pymzl.filters import MSMinLevelFilter
from pyspec.parser.pymzl.msms_finder import MSMSFinder

parser = argparse.ArgumentParser(description="store rawdata in a database")

parser.add_argument("--rawdata", help="directory containing your rawdata", required=True, type=str)
parser.add_argument("--clob", help="clob pattern to filter by", default="*.mzml", type=str)

args = parser.parse_args()

finder = MSMSFinder()

counter = 0
expression = "{}/{}".format(args.rawdata, args.clob)

convert = MZMLtoPostgresConverter()
print("looking for data in : {}".format(expression))
for file in iglob(expression, recursive=True):
    convert.convert(file)

print("processing was finished for {} files".format(counter))
