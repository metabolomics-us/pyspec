#!/usr/bin/env python
#
# Given a file with a list of InChIKeys, retrieve the spectrum count and submitter
# information associated with each InChIKey and check whether there exists a spectrum 
# submitted by the Fiehn Lab

import argparse
import pathlib
import requests

from pyspec.loader.mona import MoNALoader, MoNAQueryGenerator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=pathlib.Path)
    args = parser.parse_args()

    # read InChIKeys and export results summary
    export_file = pathlib.Path(args.file.parent, args.file.stem + '_results.csv')

    with args.file.open() as f, export_file.open('w') as fout:
      print('InChIKey,Spectrum Count,Fiehn Spectrum,Submitters', file=fout)

      for line in f:
        inchikey = line.strip()

        if inchikey:
            print(inchikey)

            # build query
            query = MoNAQueryGenerator().query_by_inchikey(inchikey)
            data = MoNALoader().query(query)

            count = len(data)
            submitters = sorted(set(x.library['library'] if x.library is not None else x.submitter['id'] for x in data))
            is_fiehnlab = any(any(y in x.lower() for y in ['@ucdavis.edu', 'fiehn', 'fahfa', 'lipidblast']) for x in submitters)

            print(inchikey, count, is_fiehnlab, *submitters, sep=',', file=fout)

      