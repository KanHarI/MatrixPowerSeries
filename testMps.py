
import scipy as sp
import numpy as np
from keras.models import Sequential

import Mps
import KerasMps

def test_mps():
	data = np.vectorize(complex)(np.random.random((1000,10,10)),np.random.random((1000,10,10)))
	labels = []
	for datum in data:
		labels.append(sp.linalg.expm(datum))
	labels = np.array(labels)
	
	test_data = np.vectorize(complex)(np.random.random((100,10,10)),np.random.random((100,10,10)))
	test_labels = []
	for datum in test_data:
		test_labels.append(sp.linalg.expm(datum))
	test_labels = np.array(test_labels)

	model = Sequential([
		KerasMps.MatrixPowerSeriesLayer(5, input_shape=(10,10))
		])

	model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

	model.fit(data, labels, epochs=10, batch_size=10)
	print(model.evaluate(test_data, test_labels, batch_size=10))
