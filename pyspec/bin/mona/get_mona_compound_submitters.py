#!/usr/bin/env python
#
# Given a file with a list of InChIKeys, retrieve the spectrum count and submitter
# information associated with each InChIKey and check whether there exists a spectrum 
# submitted by the Fiehn Lab

import argparse
import pathlib
import requests


MONA_URL = "https://mona.fiehnlab.ucdavis.edu/rest/spectra/search?query=compound.metaData=q='name==\"InChIKey\" and value==\"%s\"'"


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

        if not inchikey:
          continue

        print(inchikey)
        r = requests.get(MONA_URL % inchikey)

        if r.status_code == 200:
          data = r.json()

          count = len(data)
          submitters = sorted(set(x['library']['library'] if 'library' in x else x['submitter']['id'] for x in data))
          is_fiehnlab = any(any(y in x.lower() for y in ['@ucdavis.edu', 'fiehn', 'fahfa', 'lipidblast']) for x in submitters)

          print(inchikey, count, is_fiehnlab, *submitters, sep=',', file=fout)
        else:
          print(f'\t{r.status_code}')
      