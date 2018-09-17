
import scipy as sp
import numpy as np
from keras.models import Sequential
import keras

import KerasMps

LR = 1
DECAY = 0.25
BATCH_SIZE = 256

# func - the function approximation being tested
# layer - the layer being tested
def generic_test_scalar(func, layer, epochs=100, samples=10000, test_samples=100, matrix_size=10, degree=5):
    # Creating training dataset
    # Initialize source matrix with random elements in the [-1,-1j]X[1,1j]
    # square on the complex plane
    data = np.vectorize(complex)(
        np.random.random((samples,matrix_size,matrix_size)),
        np.random.random((samples,matrix_size,matrix_size)))*2-1-1j

    # Calculate expected results
    labels = []
    for datum in data:
        labels.append(func(datum))
    labels = np.array(labels)

    # Seprate into real and complex parts
    kdata = np.array(list(map(lambda x: [x.real, x.imag], data)))
    klabels = np.array(list(map(lambda x: [x.real, x.imag], labels)))
    
    # Craeting test dataset in the same way
    test_data = np.vectorize(complex)(
        np.random.random((test_samples,matrix_size,matrix_size)),
        np.random.random((test_samples,matrix_size,matrix_size)))*2-1-1j

    test_labels = []
    for datum in test_data:
        test_labels.append(func(datum))
    test_labels = np.array(test_labels)

    ktest_data = np.array(list(map(lambda x: [x.real, x.imag], test_data)))
    ktest_labels = np.array(list(map(lambda x: [x.real, x.imag], test_labels)))

    # Create model
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

# For some reason these converge much slower than the m2 variant
test_zero_m = lambda **kwargs: generic_test_scalar(lambda x: np.zeros(x.shape), KerasMps.MatrixMPowerSeriesLayer, **kwargs)
test_unit_m = lambda **kwargs: generic_test_scalar(lambda x: np.identity(x.shape[0]), KerasMps.MatrixMPowerSeriesLayer, **kwargs)
test_id_m = lambda **kwargs: generic_test_scalar(lambda x: x, KerasMps.MatrixMPowerSeriesLayer, **kwargs)
test_square_m = lambda **kwargs: generic_test_scalar(lambda x: np.matmul(x,x), KerasMps.MatrixMPowerSeriesLayer, **kwargs)
test_exp_m = lambda **kwargs: generic_test_scalar(lambda x: sp.linalg.expm(x), KerasMps.MatrixMPowerSeriesLayer, **kwargs)

# For some reason these converge much faster than the previous
test_zero_m2 = lambda **kwargs: generic_test_scalar(lambda x: np.zeros(x.shape), KerasMps.MatrixM2PowerSeriesLayer, **kwargs)
test_unit_m2 = lambda **kwargs: generic_test_scalar(lambda x: np.identity(x.shape[0]), KerasMps.MatrixM2PowerSeriesLayer, **kwargs)
test_id_m2 = lambda **kwargs: generic_test_scalar(lambda x: x, KerasMps.MatrixM2PowerSeriesLayer, **kwargs)
test_square_m2 = lambda **kwargs: generic_test_scalar(lambda x: np.matmul(x,x), KerasMps.MatrixM2PowerSeriesLayer, **kwargs)
test_exp_m2 = lambda **kwargs: generic_test_scalar(lambda x: sp.linalg.expm(x), KerasMps.MatrixM2PowerSeriesLayer, **kwargs)


# 
