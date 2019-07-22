from pyspec.machine.labels.generate_labels import DirectoryLabelGenerator
from pyspec.machine.model.application import XceptionModel

batchsize = 2
epochs = 500
model = XceptionModel(width=500, height=500, channels=3, plots=True, batch_size=batchsize, gpus=3)
generator = DirectoryLabelGenerator()

m = model.train("datasets/clean_dirty_full", generator, epochs=epochs)
