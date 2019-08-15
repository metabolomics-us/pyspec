import argparse
import os

from pyspec.machine.factory import MachineFactory
from pyspec.machine.util.gpu import get_gpu_count

parser = argparse.ArgumentParser(description="train a neural network")

parser.add_argument("--model", help="model you would like to train", required=False, type=str, default=None)
parser.add_argument("--dataset", help="path to your dataset", required=True, type=str)
parser.add_argument("--configuration", help="which configuration file to use", required=True, type=str)
parser.add_argument("--gpu", help="which gpu to use, by deault all will be utilized", type=int, default=-1)

args = parser.parse_args()
factory = MachineFactory(config_file=args.configuration)

model = factory.load_model(name=args.model)

if args.gpu < 0:
    factory.train(args.dataset, model=model, gpus=get_gpu_count())
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(args.gpu)
    factory.train(args.dataset, model=model, gpus=1)
