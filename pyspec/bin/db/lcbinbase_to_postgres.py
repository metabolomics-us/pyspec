import argparse
from glob import iglob

from tqdm import tqdm

from pyspec.converter.carrot_msms_dump_to_postgres import LCBinBaseToPostgresConverter

parser = argparse.ArgumentParser(description="converts a lc-binbase sample to postgres classification data")

parser.add_argument("--clob", help="your sample name patterns, as like expression", required=False, type=str,
                    default="*.msms.json*")
parser.add_argument("--directory", help="directory containing your data", required=True, type=str)

args = parser.parse_args()

converter = LCBinBaseToPostgresConverter()

expression = "{}/{}".format(args.directory, args.clob)
for file in tqdm(iglob(expression, recursive=True), desc="importing lc binbase data"):
    converter.convert(file=file, compressed=True if '.gz' in file else False)
