
import scipy as sp
import numpy as np
from keras.models import Sequential
import keras

import Mps
import KerasMps

def test_zero():
	data = np.vectorize(complex)(np.random.random((1000,10,10)),np.random.random((1000,10,10)))*2-1-1j
	labels = []
	for datum in data:
		labels.append(np.zeros((10,10)))
	labels = np.array(labels)

	kdata = np.array(list(map(lambda x: [x.real, x.imag], data)))
	klabels = np.array(list(map(lambda x: [x.real, x.imag], labels)))
	
	test_data = np.vectorize(complex)(np.random.random((100,10,10)),np.random.random((100,10,10)))*2-1-1j
	test_labels = []
	for datum in test_data:
		test_labels.append(np.zeros((10,10)))
	test_labels = np.array(test_labels)

	ktest_data = np.array(list(map(lambda x: [x.real, x.imag], test_data)))
	ktest_labels = np.array(list(map(lambda x: [x.real, x.imag], test_labels)))

	model = Sequential([
		KerasMps.MatrixPowerSeriesLayer(5, input_shape=(2,10,10))
		])

	sgd = keras.optimizers.SGD(lr=0.1, decay=0.01, momentum=0.5, nesterov=True,clipnorm=100)
	model.compile(optimizer=sgd,
              loss='mse')

	model.fit(kdata, klabels, epochs=1000, batch_size=128)
	print(model.evaluate(ktest_data, ktest_labels, batch_size=128))

	return model

def test_unit():
	data = np.vectorize(complex)(np.random.random((1000,10,10)),np.random.random((1000,10,10)))*2-1-1j
	labels = []
	for datum in data:
		labels.append(np.identity(10))
	labels = np.array(labels)

	kdata = np.array(list(map(lambda x: [x.real, x.imag], data)))
	klabels = np.array(list(map(lambda x: [x.real, x.imag], labels)))
	
	test_data = np.vectorize(complex)(np.random.random((100,10,10)),np.random.random((100,10,10)))*2-1-1j
	test_labels = []
	for datum in test_data:
		test_labels.append(np.identity(10))
	test_labels = np.array(test_labels)

	ktest_data = np.array(list(map(lambda x: [x.real, x.imag], test_data)))
	ktest_labels = np.array(list(map(lambda x: [x.real, x.imag], test_labels)))

	model = Sequential([
		KerasMps.MatrixPowerSeriesLayer(5, input_shape=(2,10,10))
		])

	sgd = keras.optimizers.SGD(lr=0.1, decay=0.01, momentum=0.5, nesterov=True,clipnorm=100)
	model.compile(optimizer=sgd,
              loss='mse')

	model.fit(kdata, klabels, epochs=1000, batch_size=128)
	print(model.evaluate(ktest_data, ktest_labels, batch_size=128))

	return model


def test_id():
	data = np.vectorize(complex)(np.random.random((1000,10,10)),np.random.random((1000,10,10)))*2-1-1j
	labels = data

	kdata = np.array(list(map(lambda x: [x.real, x.imag], data)))
	klabels = np.array(list(map(lambda x: [x.real, x.imag], labels)))
	
	test_data = np.vectorize(complex)(np.random.random((100,10,10)),np.random.random((100,10,10)))*2-1-1j
	test_labels = test_data

	ktest_data = np.array(list(map(lambda x: [x.real, x.imag], test_data)))
	ktest_labels = np.array(list(map(lambda x: [x.real, x.imag], test_labels)))

	model = Sequential([
		KerasMps.MatrixPowerSeriesLayer(5, input_shape=(2,10,10))
		])

	sgd = keras.optimizers.SGD(lr=0.1, decay=0.01, momentum=0.5, nesterov=True,clipnorm=100)
	model.compile(optimizer=sgd,
              loss='mse')

	model.fit(kdata, klabels, epochs=1000, batch_size=128)
	print(model.evaluate(ktest_data, ktest_labels, batch_size=128))

	return model


def test_square():
	data = np.vectorize(complex)(np.random.random((1000,10,10)),np.random.random((1000,10,10)))*2-1-1j
	labels = []
	for datum in data:
		labels.append(datum.dot(datum))
	labels = np.array(labels)

	kdata = np.array(list(map(lambda x: [x.real, x.imag], data)))
	klabels = np.array(list(map(lambda x: [x.real, x.imag], labels)))
	
	test_data = np.vectorize(complex)(np.random.random((100,10,10)),np.random.random((100,10,10)))*2-1-1j
	test_labels = []
	for datum in test_data:
		test_labels.append(datum.dot(datum))
	test_labels = np.array(test_labels)

	ktest_data = np.array(list(map(lambda x: [x.real, x.imag], test_data)))
	ktest_labels = np.array(list(map(lambda x: [x.real, x.imag], test_labels)))

	model = Sequential([
		KerasMps.MatrixPowerSeriesLayer(5, input_shape=(2,10,10))
		])

	sgd = keras.optimizers.SGD(lr=0.1, decay=0.01, momentum=0.5, nesterov=True,clipnorm=100)
	model.compile(optimizer=sgd,
              loss='mse')

	model.fit(kdata, klabels, epochs=1000, batch_size=16)
	print(model.evaluate(ktest_data, ktest_labels, batch_size=10))

	return model


def test_exp():
	data = np.vectorize(complex)(np.random.random((1000,10,10)),np.random.random((1000,10,10)))*2-1-1j
	labels = []
	for datum in data:
		labels.append(sp.linalg.expm(datum))
	labels = np.array(labels)

	kdata = np.array(list(map(lambda x: [x.real, x.imag], data)))
	klabels = np.array(list(map(lambda x: [x.real, x.imag], labels)))
	
	test_data = np.vectorize(complex)(np.random.random((100,10,10)),np.random.random((100,10,10)))*2-1-1j
	test_labels = []
	for datum in test_data:
		test_labels.append(sp.linalg.expm(datum))
	test_labels = np.array(test_labels)

	ktest_data = np.array(list(map(lambda x: [x.real, x.imag], test_data)))
	ktest_labels = np.array(list(map(lambda x: [x.real, x.imag], test_labels)))

	model = Sequential([
		KerasMps.MatrixPowerSeriesLayer(5, input_shape=(2,10,10))
		])

	sgd = keras.optimizers.SGD(lr=0.1, decay=0.01, momentum=0.5, nesterov=True,clipnorm=100)
	model.compile(optimizer=sgd,
              loss='mse')

	model.fit(kdata, klabels, epochs=1000, batch_size=16)
	print(model.evaluate(ktest_data, ktest_labels, batch_size=10))

	return model
