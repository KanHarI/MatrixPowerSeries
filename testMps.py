
import scipy as sp
import numpy as np
from keras.models import Sequential
import keras

import KerasMps

LR = 0.5
DECAY = 0.1
BATCH_SIZE = 256

def generic_test_scalar(func, layer, epochs=1000, samples=1000, test_samples=100, matrix_size=10, degree=5):
    data = np.vectorize(complex)(
        np.random.random((samples,matrix_size,matrix_size)),
        np.random.random((samples,matrix_size,matrix_size)))*2-1-1j

    labels = []
    for datum in data:
        labels.append(func(datum))
    labels = np.array(labels)

    kdata = np.array(list(map(lambda x: [x.real, x.imag], data)))
    klabels = np.array(list(map(lambda x: [x.real, x.imag], labels)))
    
    test_data = np.vectorize(complex)(
        np.random.random((test_samples,matrix_size,matrix_size)),
        np.random.random((test_samples,matrix_size,matrix_size)))*2-1-1j

    test_labels = []
    for datum in test_data:
        test_labels.append(func(datum))
    test_labels = np.array(test_labels)

    ktest_data = np.array(list(map(lambda x: [x.real, x.imag], test_data)))
    ktest_labels = np.array(list(map(lambda x: [x.real, x.imag], test_labels)))

    model = Sequential([
        layer(degree, input_shape=(2,matrix_size,matrix_size))
        ])

    opt = keras.optimizers.RMSprop(lr=LR, decay=DECAY)
    model.compile(optimizer=opt,
                    loss='logcosh')
    model.fit(kdata, klabels, epochs=epochs, batch_size=BATCH_SIZE)

    return (model.evaluate(ktest_data, ktest_labels, batch_size=BATCH_SIZE), model)



test_zero = lambda **kwargs: generic_test_scalar(lambda x: np.zeros(x.shape), KerasMps.MatrixPowerSeriesLayer, **kwargs)
test_unit = lambda **kwargs: generic_test_scalar(lambda x: np.identity(x.shape[0]), KerasMps.MatrixPowerSeriesLayer, **kwargs)
test_id = lambda **kwargs: generic_test_scalar(lambda x: x, KerasMps.MatrixPowerSeriesLayer, **kwargs)
test_square = lambda **kwargs: generic_test_scalar(lambda x: np.matmul(x,x), KerasMps.MatrixPowerSeriesLayer, **kwargs)
test_exp = lambda **kwargs: generic_test_scalar(lambda x: sp.linalg.expm(x), KerasMps.MatrixPowerSeriesLayer, **kwargs)

test_zero_m = lambda **kwargs: generic_test_scalar(lambda x: np.zeros(x.shape), KerasMps.MatrixMPowerSeriesLayer, **kwargs)
test_unit_m = lambda **kwargs: generic_test_scalar(lambda x: np.identity(x.shape[0]), KerasMps.MatrixMPowerSeriesLayer, **kwargs)
test_id_m = lambda **kwargs: generic_test_scalar(lambda x: x, KerasMps.MatrixMPowerSeriesLayer, **kwargs)
test_square_m = lambda **kwargs: generic_test_scalar(lambda x: np.matmul(x,x), KerasMps.MatrixMPowerSeriesLayer, **kwargs)
test_exp_m = lambda **kwargs: generic_test_scalar(lambda x: sp.linalg.expm(x), KerasMps.MatrixMPowerSeriesLayer, **kwargs)
