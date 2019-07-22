###
# exports bins as MSP file

import argparse

from pyspec.msp.writer import MSP

parser = argparse.ArgumentParser(description='This script will export a given list of Bins as MSP')
parser.add_argument('bins', type=int, nargs='+',
                    help='several bin ids')

args = parser.parse_args()

msp = MSP()

for x in args.bins:
    print(msp.from_bin(x))
    print()
