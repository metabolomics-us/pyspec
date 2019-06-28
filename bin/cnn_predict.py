from keras_preprocessing.image import load_img
from pandas import DataFrame
from shutil import copyfile

import matplotlib.pyplot as plt

batchsize = 2
from pyspec.machine.model.cnn import ClassificationModel
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

model = ClassificationModel(width=500, height=500, channels=3, plots=True, batch_size=batchsize)


def callback(file, classname):
    ##  print("{}:{}".format(file, classname))
    ##  plt.figure(figsize=(6, 6))
    ##  img = load_img("{}/{}".format("datasets/clean_dirty_full/test", file), target_size=(500, 500))
    ##  plt.imshow(img)
    ##  plt.title(file + '\n(' + "{}".format(classname) + ')')
    ##  plt.show()

    if classname == 0:
        classname = "clean"
    else:
        classname = "dirty"

    if not os.path.exists("{}/{}".format("datasets/clean_dirty_full/sorted", classname)):
        os.mkdir("{}/{}".format("datasets/clean_dirty_full/sorted", classname))
    copyfile("{}/{}".format("datasets/clean_dirty_full/test", file),
             "{}/{}/{}".format("datasets/clean_dirty_full/sorted", classname, file))


model.predict_from_directory(input="datasets/clean_dirty_full", dict="datasets/clean_dirty_full/test",
                             callback=callback)
