from keras_preprocessing.image import load_img
from pandas import DataFrame
from shutil import copyfile

import matplotlib.pyplot as plt

from pyspec.machine.model.Xception import XceptionModel
from pyspec.machine.model.simple_cnn import PoolingCNNModel, SimpleCNNModel

batchsize = 2
from pyspec.machine.model.cnn import CNNClassificationModel
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

model = XceptionModel(width=500, height=500, channels=3, plots=True, batch_size=batchsize)


def callback(file, classname):
    if classname == 0:
        classname = "clean"
    else:
        classname = "dirty"

    print("{} is {}".format(file, classname))
    os.makedirs("{}/{}".format("datasets/clean_dirty_full/sorted", classname), exist_ok=True)
    copyfile("{}/{}".format("datasets/clean_dirty_full/test", file),
             "{}/{}/{}".format("datasets/clean_dirty_full/sorted", classname, file))


model.predict_from_directory(input="datasets/clean_dirty_full", dict="datasets/clean_dirty_full/test",
                             callback=callback)
