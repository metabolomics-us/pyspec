#!/usr/bin/env python

"""
Pulls all bins and counts associated with a given set of species
"""

import argparse
import collections
from typing import List

import pandas as pd
import pymongo
import tqdm

MONGO_HOST = 'venus.fiehnlab.ucdavis.edu'
MONGO_PORT = 27017
MONGO_DB = 'binvestigate'


def pull_bins_by_species(species: List[str]):
    client = pymongo.MongoClient(MONGO_HOST, MONGO_PORT)
    db = client[MONGO_DB]

    # Count species+organ+bin combinations
    cursor = db.annotations.find({'species': {'$in': species}})
    count = collections.defaultdict(int)

    for x in tqdm.tqdm(cursor):
        count[(x['species'], x['organ'], x['annotations'][0]['binid'], x['annotations'][0]['bin'])] += 1

    # Return results as a data frame
    columns = ['species', 'organ', 'binid', 'bin', 'count']
    data = [(*k, v) for k, v in count.items()]

    return pd.DataFrame(data, columns=columns)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='downloads all bins and counts associated a given set of species')
    parser.add_argument('species', nargs='+', help='species list')
    parser.add_argument('-o', '--output', required=True, help='csv output filename')
    args = parser.parse_args()

    df = pull_bins_by_species(args.species)
    df.to_csv(args.output, index=False)
