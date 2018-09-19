
import scipy as sp
import numpy as np
from keras.models import Sequential
import keras
import keras.backend as K
import tensorflow as tf
import copy

from KerasMps import MPS, MMPS, MM2PS, MchMPS, MchMMPS, MchMM2PS

LR = 1
DECAY = 0.01
MOMENTUM = 0.9

OPTIMIZER = keras.optimizers.SGD(lr=LR, decay=DECAY, momentum=MOMENTUM)

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
def generic_test_scalar(func, layer, epochs=100, samples=4000, test_samples=100, matrix_size=6, degree=5, batch_size=256):
    # Create training and test datasets
    train_data, train_labels = generate_samples(func, matrix_size, samples)
    test_data, test_labels = generate_samples(func, matrix_size, test_samples)
    
    # Create model
    model = Sequential([
        layer(degree=degree, input_shape=(2,matrix_size,matrix_size))
        ])

    model.compile(optimizer=OPTIMIZER,
                    loss='logcosh')
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)

    return (model.evaluate(test_data, test_labels, batch_size=batch_size), model)



zero = lambda x: np.zeros(x.shape)
unit = lambda x: np.identity(x.shape[0])
identity = lambda x: x
square = lambda x: np.matmul(x,x)
cube = lambda x: np.matmul(x,square(x))
expm = sp.linalg.expm

# This function was designed as a test for the M2 variant
# It cannot be approximated by matrix multiplication from only one side
def bold(x):
    x = copy.deepcopy(x)
    # random numbers
    x[0,:] *= 3+1j
    x[:,0] *= -2-3j
    return x


test_zero = lambda **kwargs: generic_test_scalar(zero, MPS, **kwargs)
test_unit = lambda **kwargs: generic_test_scalar(unit, MPS, **kwargs)
test_id = lambda **kwargs: generic_test_scalar(identity, MPS, **kwargs)
test_square = lambda **kwargs: generic_test_scalar(square, MPS, **kwargs)
test_cube = lambda **kwargs: generic_test_scalar(cube, MPS, **kwargs)
test_exp = lambda **kwargs: generic_test_scalar(expm, MPS, **kwargs)
test_bold = lambda **kwargs: generic_test_scalar(bold, MPS, **kwargs)

test_zero_m = lambda **kwargs: generic_test_scalar(zero, MMPS, **kwargs)
test_unit_m = lambda **kwargs: generic_test_scalar(unit, MMPS, **kwargs)
test_id_m = lambda **kwargs: generic_test_scalar(identity, MMPS, **kwargs)
test_square_m = lambda **kwargs: generic_test_scalar(square, MMPS, **kwargs)
test_cube_m = lambda **kwargs: generic_test_scalar(cube, MMPS, **kwargs)
test_exp_m = lambda **kwargs: generic_test_scalar(expm, MMPS, **kwargs)
test_bold_m = lambda **kwargs: generic_test_scalar(bold, MMPS, **kwargs)

test_zero_m2 = lambda **kwargs: generic_test_scalar(zero, MM2PS, **kwargs)
test_unit_m2 = lambda **kwargs: generic_test_scalar(unit, MM2PS, **kwargs)
test_id_m2 = lambda **kwargs: generic_test_scalar(identity, MM2PS, **kwargs)
test_square_m2 = lambda **kwargs: generic_test_scalar(square, MM2PS, **kwargs)
test_cube_m2 = lambda **kwargs: generic_test_scalar(cube, MM2PS, **kwargs)
test_exp_m2 = lambda **kwargs: generic_test_scalar(expm, MM2PS, **kwargs)
test_bold_m2 = lambda **kwargs: generic_test_scalar(bold, MM2PS, **kwargs)


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
                            input_channels=5,
                            epochs=100,
                            samples=1000,
                            test_samples=100,
                            matrix_size=6,
                            degree=5,
                            batch_size=64):
    train_data, train_labels = multichannel_generate_samples(func, matrix_size, input_channels, samples)
    test_data, test_labels = multichannel_generate_samples(func, matrix_size, input_channels, test_samples)

    model = Sequential([
        layer(
            degree=degree,
            out_channels=out_channels,
            input_shape=(input_channels,2,matrix_size,matrix_size)) # input_shape=[k,l,m,n]
        # output_shape: [j,k,l,m,n]
        ])

    model.compile(optimizer=OPTIMIZER,
                    loss='logcosh')
    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)

    return (model.evaluate(test_data, test_labels, batch_size=batch_size), model)

zero_id_unit = lambda x: np.array([zero(x),unit(x),identity(x)])
square_exp_cube = lambda x: np.array([square(x),expm(x),cube(x)])
bold_2 = lambda x: np.array([bold(x),bold(x)])

def make_test_multi(func, layer, out_channels, **kwargs):
    return lambda **kwargs: generic_multichannel_test(
                                func=func,
                                layer=layer,
                                out_channels=out_channels,
                                **kwargs)

test_multichannel = make_test_multi(zero_id_unit, MchMPS, 3)
test_multichannel_nonlin = make_test_multi(square_exp_cube, MchMPS, 3)
test_multichannel_bold = make_test_multi(bold_2, MchMPS, 2)

test_multichannel_m = make_test_multi(zero_id_unit, MchMMPS, 3)
test_multichannel_nonlin_m = make_test_multi(square_exp_cube, MchMMPS, 3)
test_multichannel_bold_m = make_test_multi(bold_2, MchMMPS, 2)

test_multichannel_m2 = make_test_multi(zero_id_unit, MchMM2PS, 3)
test_multichannel_nonlin_m2 = make_test_multi(square_exp_cube, MchMM2PS, 3)
test_multichannel_bold_m2 = make_test_multi(bold_2, MchMM2PS, 2)
