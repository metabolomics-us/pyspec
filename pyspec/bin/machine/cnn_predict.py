import argparse
from shutil import copyfile

from pyspec.machine.factory import MachineFactory

import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

parser = argparse.ArgumentParser(description="train a neural network")
parser.add_argument("--dataset", help="path to your dataset", required=True, type=str)
parser.add_argument("--predict", help="path to the directory containing the data you want to predict", required=True,
                    type=str)
parser.add_argument("--configuration", help="which configuration file to use", required=True, type=str)
parser.add_argument("--gpu", help="which gpu to use", type=int, default=-1)

parser.add_argument("--model", help="model you would like to predict", required=False, type=str, default=None)

args = parser.parse_args()
factory = MachineFactory(config_file=args.configuration)

model = factory.load_model(name=args.model)


def callback(file, classname):
    if classname == 0:
        classname = "clean"
    else:
        classname = "dirty"

    print("{} is {}".format(file, classname))
    os.makedirs("{}/{}".format("datasets/clean_dirty_full/sorted", classname), exist_ok=True)
    copyfile("{}/{}".format("datasets/clean_dirty_full/test", file),
             "{}/{}/{}".format("datasets/clean_dirty_full/sorted", classname, file))


if args.gpu < 0:
    # do this on cpu
    model.predict_from_directory(input=args.dataset, dict=args.predict,
                                 callback=callback)
else:
    model.predict_from_directory(input=args.dataset, dict=args.predict,
                                 callback=callback)
