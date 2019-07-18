from pyspec.machine.labels.generate_labels import DirectoryLabelGenerator
from pyspec.machine.model.Xception import XceptionModel
from pyspec.machine.model.cnn import CNNClassificationModel
from pyspec.machine.model.simple_cnn import SimpleCNNModel

batchsize = 2
epochs = 50
model = XceptionModel(width=500, height=500, channels=3, plots=True, batch_size=batchsize)
generator = DirectoryLabelGenerator()

m = model.train("datasets/clean_dirty_full", generator, epochs=epochs)
