import argparse
from shutil import copyfile

from pyspec.machine.factory import MachineFactory

import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pyspec.machine.share.s3 import S3Share

parser = argparse.ArgumentParser(description="train a neural network")
parser.add_argument("--dataset", help="name of your dataset", required=True, type=str)
parser.add_argument("--predict", help="path to the directory containing the data you want to predict", required=True,
                    type=str)
parser.add_argument("--configuration", help="which configuration file to use", required=False, type=str,
                    default="machine.ini")
parser.add_argument("--gpu", help="which gpu to use", type=int, default=-1)

parser.add_argument("--model", help="model you would like to predict", required=False, type=str, default=None)

args = parser.parse_args()
factory = MachineFactory(config_file=args.configuration)

model = factory.load_model(name=args.model)


def callback(file, classname, full_path: str):
    if classname == 0:
        classname = "clean"
    else:
        classname = "dirty"

    print("{} is {}".format(file, classname))
    os.makedirs("{}/{}".format("datasets/{}/sorted".format(args.dataset), classname), exist_ok=True)
    copyfile(full_path, "{}/{}/{}".format("datasets/{}/sorted".format(args.dataset), classname, file))

# download trained models for a given dataset
share = S3Share(read_only=True)
share.retrieve(args.dataset, force=False)

if args.gpu < 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # do this on cpu
    model.predict_from_directory(input="datasets/{}".format(args.dataset), dict=args.predict,
                             callback=callback)
else:
    model.predict_from_directory(input="datasets/{}".format(args.dataset), dict=args.predict,
                             callback=callback)
