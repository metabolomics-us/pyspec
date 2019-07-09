##
# simple bin script to encode all files in a directory into images
import multiprocessing
from glob import glob1, glob, iglob
from xml.etree.ElementTree import ParseError

from pathlib import Path

import argparse
import os
from pymzml.spec import Spectrum

from pyspec.machine.spectra import Encoder
from pyspec.parser.pymzl.filters import MSMinLevelFilter
from pyspec.parser.pymzl.msms_finder import MSMSFinder

parser = argparse.ArgumentParser(description="encode rawdata to image files")

parser.add_argument("--rawdata", help="directory containing your rawdata", required=True, type=str)
parser.add_argument("--destination", help="directory where you want to store the data", required=True, type=str)
parser.add_argument("--group", action="store", help="do you want to group the encoded images by filename", default=True)
parser.add_argument("--dimension", help="the size of the image to be generated", default=500)
parser.add_argument("--min_mz", help="the minimum mass", default=0, type=int)
parser.add_argument("--max_mz", help="the maximum mass", default=2000, type=int)
parser.add_argument("--max_intensity", help="the maximum mass", default=10000, type=int)

parser.add_argument("--clob", help="clob pattern to filter by", default="*.mzml", type=str)

args = parser.parse_args()

finder = MSMSFinder()

counter = 0
expression = "{}/{}".format(args.rawdata, args.clob)

print("looking for data in : {}".format(expression))
for file in iglob(expression, recursive=True):

    counter = counter + 1
    if args.group is True:
        encoder = Encoder(intensity_max=args.max_intensity, min_mz=args.min_mz, max_mz=args.max_mz,
                          directory="{}/{}".format(args.destination, Path(file).name))
    else:
        encoder = Encoder(intensity_max=args.max_intensity, min_mz=args.min_mz, max_mz=args.max_mz,
                          directory="{}".format(args.destination))

    data = []


    def callback(msms: Spectrum, file_name: str):
        """
        builds our data list
        :param msms:
        :param file_name:
        :return:
        """
        data.append(msms.convert(msms))


    try:
        finder.locate(msmsSource=file, callback=callback, filters=[MSMinLevelFilter(2)])

        if len(data) > 0:
            from joblib import Parallel, delayed

            Parallel(n_jobs=multiprocessing.cpu_count())(delayed(encoder.encode)(x) for x in data)
    except ParseError:
        print("ignoring file: {}, due to format errors!".format(file))

print("processing was finished for {} files".format(counter))
