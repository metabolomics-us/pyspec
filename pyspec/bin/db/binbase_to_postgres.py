import argparse

parser = argparse.ArgumentParser(description="converts a binbase sample to postgres classification data")

parser.add_argument("--pattern", help="your sample name patterns, as like expression", required=False, type=str,
                    default="%%")

args = parser.parse_args()

from pyspec.converter.binbase_to_postgres import BinBasetoPostgresConverter

converter = BinBasetoPostgresConverter()
converter.convert(pattern=args.pattern)
