
import scipy as sp
import numpy as np
from keras.models import Sequential
import keras

import KerasMps

LR = 1
DECAY = 0.1
MOMENTUM = 0.5

def test_zero(epochs=1000):
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

    sgd = keras.optimizers.SGD(lr=LR, decay=DECAY, momentum=MOMENTUM, nesterov=True,clipnorm=100)
    model.compile(optimizer=sgd,
              loss='mse')

    model.fit(kdata, klabels, epochs=epochs, batch_size=128)
    print(model.evaluate(ktest_data, ktest_labels, batch_size=128))

    return model

def test_unit(epochs=1000):
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

    sgd = keras.optimizers.SGD(lr=LR, decay=DECAY, momentum=MOMENTUM, nesterov=True,clipnorm=100)
    model.compile(optimizer=sgd,
              loss='mse')

    model.fit(kdata, klabels, epochs=epochs, batch_size=128)
    print(model.evaluate(ktest_data, ktest_labels, batch_size=128))

    return model


def test_id(epochs=1000):
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

    sgd = keras.optimizers.SGD(lr=LR, decay=DECAY, momentum=MOMENTUM, nesterov=True,clipnorm=100)
    model.compile(optimizer=sgd,
              loss='mse')

    model.fit(kdata, klabels, epochs=epochs, batch_size=128)
    print(model.evaluate(ktest_data, ktest_labels, batch_size=128))

    return model


def test_square(epochs=1000):
    data = np.vectorize(complex)(np.random.random((1000,10,10)),np.random.random((1000,10,10)))*2-1-1j
    labels = []
    for datum in data:
        labels.append(np.matmul(datum, datum))
    labels = np.array(labels)

    kdata = np.array(list(map(lambda x: [x.real, x.imag], data)))
    klabels = np.array(list(map(lambda x: [x.real, x.imag], labels)))
    
    test_data = np.vectorize(complex)(np.random.random((100,10,10)),np.random.random((100,10,10)))*2-1-1j
    test_labels = []
    for datum in test_data:
        test_labels.append(np.matmul(datum, datum))
    test_labels = np.array(test_labels)

    ktest_data = np.array(list(map(lambda x: [x.real, x.imag], test_data)))
    ktest_labels = np.array(list(map(lambda x: [x.real, x.imag], test_labels)))

    model = Sequential([
        KerasMps.MatrixPowerSeriesLayer(5, input_shape=(2,10,10))
        ])

    sgd = keras.optimizers.SGD(lr=LR, decay=DECAY, momentum=MOMENTUM, nesterov=True,clipnorm=100)
    model.compile(optimizer=sgd,
              loss='mse')

    model.fit(kdata, klabels, epochs=epochs, batch_size=128)
    print(model.evaluate(ktest_data, ktest_labels, batch_size=128))

    return model


def test_exp(epochs=1000):
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

    sgd = keras.optimizers.SGD(lr=LR, decay=DECAY, momentum=MOMENTUM, nesterov=True,clipnorm=100)
    model.compile(optimizer=sgd,
              loss='mse')

    model.fit(kdata, klabels, epochs=epochs, batch_size=128)
    print(model.evaluate(ktest_data, ktest_labels, batch_size=128))

    return model


# Matrix coefficients tests
def test_zero_m(epochs=1000):
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
        KerasMps.MatrixMPowerSeriesLayer(5, input_shape=(2,10,10))
        ])

    sgd = keras.optimizers.SGD(lr=LR, decay=DECAY, momentum=MOMENTUM, nesterov=True,clipnorm=100)
    model.compile(optimizer=sgd,
              loss='mse')

    model.fit(kdata, klabels, epochs=epochs, batch_size=128)
    print(model.evaluate(ktest_data, ktest_labels, batch_size=128))

    return model

def test_unit_m(epochs=1000):
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
        KerasMps.MatrixMPowerSeriesLayer(5, input_shape=(2,10,10))
        ])

    sgd = keras.optimizers.SGD(lr=LR, decay=DECAY, momentum=MOMENTUM, nesterov=True,clipnorm=100)
    model.compile(optimizer=sgd,
              loss='mse')

    model.fit(kdata, klabels, epochs=epochs, batch_size=128)
    print(model.evaluate(ktest_data, ktest_labels, batch_size=128))

    return model


def test_id_m(epochs=1000):
    data = np.vectorize(complex)(np.random.random((1000,10,10)),np.random.random((1000,10,10)))*2-1-1j
    labels = data

    kdata = np.array(list(map(lambda x: [x.real, x.imag], data)))
    klabels = np.array(list(map(lambda x: [x.real, x.imag], labels)))
    
    test_data = np.vectorize(complex)(np.random.random((100,10,10)),np.random.random((100,10,10)))*2-1-1j
    test_labels = test_data

    ktest_data = np.array(list(map(lambda x: [x.real, x.imag], test_data)))
    ktest_labels = np.array(list(map(lambda x: [x.real, x.imag], test_labels)))

    model = Sequential([
        KerasMps.MatrixMPowerSeriesLayer(5, input_shape=(2,10,10))
        ])

    sgd = keras.optimizers.SGD(lr=LR, decay=DECAY, momentum=MOMENTUM, nesterov=True,clipnorm=100)
    model.compile(optimizer=sgd,
              loss='mse')

    model.fit(kdata, klabels, epochs=epochs, batch_size=128)
    print(model.evaluate(ktest_data, ktest_labels, batch_size=128))

    return model


def test_square_m(epochs=1000):
    data = np.vectorize(complex)(np.random.random((1000,10,10)),np.random.random((1000,10,10)))*2-1-1j
    labels = []
    for datum in data:
        labels.append(np.matmul(datum, datum))
    labels = np.array(labels)

    kdata = np.array(list(map(lambda x: [x.real, x.imag], data)))
    klabels = np.array(list(map(lambda x: [x.real, x.imag], labels)))
    
    test_data = np.vectorize(complex)(np.random.random((100,10,10)),np.random.random((100,10,10)))*2-1-1j
    test_labels = []
    for datum in test_data:
        test_labels.append(np.matmul(datum, datum))
    test_labels = np.array(test_labels)

    ktest_data = np.array(list(map(lambda x: [x.real, x.imag], test_data)))
    ktest_labels = np.array(list(map(lambda x: [x.real, x.imag], test_labels)))

    model = Sequential([
        KerasMps.MatrixMPowerSeriesLayer(5, input_shape=(2,10,10))
        ])

    sgd = keras.optimizers.SGD(lr=LR, decay=DECAY, momentum=MOMENTUM, nesterov=True,clipnorm=100)
    model.compile(optimizer=sgd,
              loss='mse')

    model.fit(kdata, klabels, epochs=epochs, batch_size=128)
    print(model.evaluate(ktest_data, ktest_labels, batch_size=128))

    return model


def test_exp_m(epochs=1000):
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
        KerasMps.MatrixMPowerSeriesLayer(5, input_shape=(2,10,10))
        ])

    sgd = keras.optimizers.SGD(lr=LR, decay=DECAY, momentum=MOMENTUM, nesterov=True,clipnorm=100)
    model.compile(optimizer=sgd,
              loss='mse')

    model.fit(kdata, klabels, epochs=epochs, batch_size=128)
    print(model.evaluate(ktest_data, ktest_labels, batch_size=128))

    return model
