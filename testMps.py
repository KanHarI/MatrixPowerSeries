
import scipy as sp
import numpy as np
from keras.models import Sequential
import keras

import KerasMps

LR = 1
DECAY = 0.25

def generate_samples(func, matrix_size, samples):
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
    ridata = np.array(list(map(lambda x: [x.real, x.imag], data)))
    rilabels = np.array(list(map(lambda x: [x.real, x.imag], labels)))

    return (ridata,rilabels)

# func - the function approximation being tested
# layer - the layer being tested
def generic_test_scalar(func, layer, epochs=100, samples=5000, test_samples=100, matrix_size=6, degree=4, batch_size=256):
    # Create training and test datasets
    train_data, train_labels = generate_samples(func, matrix_size, samples)
    test_data, test_labels = generate_samples(func, matrix_size, test_samples)
    
    # Create model
    model = Sequential([
        layer(degree=degree, input_shape=(2,matrix_size,matrix_size))
        ])

    opt = keras.optimizers.RMSprop(lr=LR, decay=DECAY)
    model.compile(optimizer=opt,
                    loss='logcosh')
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)

    return (model.evaluate(test_data, test_labels, batch_size=batch_size), model)



zero = lambda x: np.zeros(x.shape)
unit = lambda x: np.identity(x.shape[0])
identity = lambda x: x
square = lambda x: np.matmul(x,x)
expm = sp.linalg.expm


test_zero = lambda **kwargs: generic_test_scalar(zero, KerasMps.MatrixPowerSeriesLayer, **kwargs)
test_unit = lambda **kwargs: generic_test_scalar(unit, KerasMps.MatrixPowerSeriesLayer, **kwargs)
test_id = lambda **kwargs: generic_test_scalar(identity, KerasMps.MatrixPowerSeriesLayer, **kwargs)
test_square = lambda **kwargs: generic_test_scalar(square, KerasMps.MatrixPowerSeriesLayer, **kwargs)
test_exp = lambda **kwargs: generic_test_scalar(expm, KerasMps.MatrixPowerSeriesLayer, **kwargs)

# For some reason these converge much slower than the m2 variant
test_zero_m = lambda **kwargs: generic_test_scalar(zero, KerasMps.MatrixMPowerSeriesLayer, **kwargs)
test_unit_m = lambda **kwargs: generic_test_scalar(unit, KerasMps.MatrixMPowerSeriesLayer, **kwargs)
test_id_m = lambda **kwargs: generic_test_scalar(identity, KerasMps.MatrixMPowerSeriesLayer, **kwargs)
test_square_m = lambda **kwargs: generic_test_scalar(square, KerasMps.MatrixMPowerSeriesLayer, **kwargs)
test_exp_m = lambda **kwargs: generic_test_scalar(expm, KerasMps.MatrixMPowerSeriesLayer, **kwargs)

# For some reason these converge much faster than the previous
test_zero_m2 = lambda **kwargs: generic_test_scalar(zero, KerasMps.MatrixM2PowerSeriesLayer, **kwargs)
test_unit_m2 = lambda **kwargs: generic_test_scalar(unit, KerasMps.MatrixM2PowerSeriesLayer, **kwargs)
test_id_m2 = lambda **kwargs: generic_test_scalar(identity, KerasMps.MatrixM2PowerSeriesLayer, **kwargs)
test_square_m2 = lambda **kwargs: generic_test_scalar(square, KerasMps.MatrixM2PowerSeriesLayer, **kwargs)
test_exp_m2 = lambda **kwargs: generic_test_scalar(expm, KerasMps.MatrixM2PowerSeriesLayer, **kwargs)


# convention:
# i is batch size
# j is output channels
# k is input channels
# l is 0/1 - real/complex part
# m,n are elements of the input matrix
# o - degree of polynomial
def multichannel_generate_samples(func, matrix_size, input_channels, samples):

    # Initialize source matrix with random elements in the [-1,-1j]X[1,1j]
    # square on the complex plane
    data = np.vectorize(complex)(
        np.random.random((samples, input_channels, matrix_size, matrix_size)),
        np.random.random((samples, input_channels, matrix_size, matrix_size)))*2-1-1j

    labels = []
    for channel in data:
        f_input_channels = []
        for datum in channel:
            f_input_channels.append(func(datum))
        labels.append(f_input_channels)
    labels = np.array(labels)

    ridata = np.einsum('likmn->iklmn', np.array([data.real, data.imag]))
    rilabels = np.einsum('likjmn->ijklmn', np.array([labels.real, labels.imag]))

    return (ridata,rilabels)

def generic_multichannel_test(func,
                            layer,
                            out_channels,
                            input_channels=7,
                            epochs=100,
                            samples=1000,
                            test_samples=100,
                            matrix_size=6,
                            degree=4,
                            batch_size=256):
    train_data, train_labels = multichannel_generate_samples(func, matrix_size, input_channels, samples)
    test_data, test_labels = multichannel_generate_samples(func, matrix_size, input_channels, test_samples)

    model = Sequential([
        layer(
            degree=degree,
            out_channels=out_channels,
            input_shape=(input_channels,2,matrix_size,matrix_size)) # input_shape=[k,l,m,n]
        # output_shape: [j,k,l,m,n]
        ])

    opt = keras.optimizers.RMSprop(lr=LR, decay=DECAY)
    model.compile(optimizer=opt,
                    loss='logcosh')
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)

    return (model.evaluate(test_data, test_labels, batch_size=batch_size), model)

zero_unit_id_square = lambda x: np.array([zero(x),unit(x),identity(x),square(x),expm(x)])

test_multichannel = lambda **kwargs: generic_multichannel_test(
    func=zero_unit_id_square,
    layer=KerasMps.MultichannelMatrixPowerSeriesLayer,
    out_channels=5,
    **kwargs)

test_multichannel_m = lambda **kwargs: generic_multichannel_test(
    func=zero_unit_id_square,
    layer=KerasMps.MultichannelMatrixMPowerSeriesLayer,
    out_channels=5,
    **kwargs)

test_multichannel_m2 = lambda **kwargs: generic_multichannel_test(
    func=zero_unit_id_square,
    layer=KerasMps.MultichannelMatrixMPowerSeriesLayer,
    out_channels=5,
    **kwargs)
