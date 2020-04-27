import argparse
import os

from pyspec.machine.factory import MachineFactory
from pyspec.machine.share.s3 import S3Share
from pyspec.machine.util.gpu import get_gpu_count

parser = argparse.ArgumentParser(description="converts a dataset to postgres classification data")

parser.add_argument("--dataset", help="name of your dataset", required=True, type=str)

args = parser.parse_args()

from pyspec.converter.dataset_to_postgres import DatesetToPostgresConverter

converter = DatesetToPostgresConverter()
converter.convert_dataset(dataset=args.dataset)
