from pyspec.machine.labels.generate_labels import DirectoryLabelGenerator
from pyspec.machine.model.cnn import CNNClassificationModel

batchsize = 2
epochs = 50
model = CNNClassificationModel(width=500, height=500, channels=3, plots=True, batch_size=batchsize)
generator = DirectoryLabelGenerator()

m = model.train("datasets/clean_dirty_full", generator, epochs=epochs)
