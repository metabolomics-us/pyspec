import argparse

from pyspec.machine.factory import MachineFactory
from pyspec.machine.util.gpu import get_gpu_count

parser = argparse.ArgumentParser(description="train a neural network")

parser.add_argument("--dataset", help="path to your dataset", required=True, type=str)
parser.add_argument("--configuration", help="which configuration file to use", required=True, type=str)

args = parser.parse_args()
factory = MachineFactory(config_file=args.configuration)

model = factory.load_model()
factory.train(args.dataset, model=model, gpus=get_gpu_count())
