
from keras.engine.topology import Layer
from keras import backend as K
import tensorflow as tf
import numpy as np
import random
import math

import ComplexTensor as ct

def factorial_decaying_random_init(shape):
    # Decay by one over factorial, inspired by the taylor expansion of the exponent function
    res = np.zeros(shape)
    for i in range(shape[0]):
        radius = random.random() / math.factorial(i)
        theta = random.random() * 2 * math.pi
        res[0] = radius*math.cos(theta)
        res[1] = radius*math.sin(theta)

    return res

# This layer represets a power series
# a_0*I+a_1*X+a_2*X^2+...+a_n*X^n
# Where X is a complex input matrix, and the coefficients a_n are complex.
# The optimized weights of this layer are the coefficients.
# Input shape: [?(batch_size), 2(0=real\1=imag), n, n] (n is the size of input matrices)
# Output shape: same as input
class MatrixPowerSeriesLayer(Layer):
    def __init__(self, degree, **kwrags):
        assert degree > 1
        self.degree = degree
        super().__init__(**kwrags)


    def build(self, input_shape):
        self.coefficients = self.add_weight(name='coefficients',
                                            shape=(self.degree,2),
                                            initializer=factorial_decaying_random_init,
                                            trainable=True)
        
        self.coefficients=ct.ComplexTensor(self.coefficients, split_axis=-1)

        self.unit = K.eye(input_shape[2])
        super().build(input_shape)

    def call(self, x):
        # Extract the real and imaginary parts of the input matrix
        x = ct.ComplexTensor(x)

        # tmp is used as the matrix raised to the n^th power
        tmp = ct.ComplexTensor.unit_like(x)

        # Initialize with the zeorth power of the series
        res = tmp*self.coefficients[0]
        
        for i in range(1, self.degree):
            # Calculate a raise by one of the current power of the matrix
            # Remainder: (a+bj)(c+dj)=(ac-bd)+(ad+bc)j, and the same 
            # formulas hold for matrices
            # new_tmp_real = tf.einsum('ijk,ikl->ijl', tmp_real, x_real) - tf.einsum('ijk,ikl->ijl', tmp_imag, x_imag)
            # tmp_imag = tf.einsum('ijk,ikl->ijl', tmp_real, x_imag) + tf.einsum('ijk,ikl->ijl', tmp_imag, x_real)
            # tmp_real = new_tmp_real
            tmp = ct.compEinsum('ijk,ikl->ijl', tmp, x)

            # Update the result with the current element of the power series
            res += tmp*self.coefficients[i]

        # Unite real and complex parts
        res = tf.stack([res.real, res.imag], axis=1)
        return res

    def compute_output_shape(self, input_shape):
        return input_shape
MPS = MatrixPowerSeriesLayer



def factorial_decaying_random_initM(shape):
    res = np.zeros(shape)
    for i in range(shape[0]): # degree
        # initiate matrix as a random multiple of the unit matrix
        radius = random.random() / math.factorial(i)
        theta = random.random() * 2 * math.pi
        res[i,0,:,:] = radius * math.cos(theta) * np.identity(shape[2])
        res[i,1,:,:] = radius * math.sin(theta) * np.identity(shape[2])
        for k in range(shape[2]):
            for l in range(shape[3]):
                # Add noise to initial matrix
                radius = random.random() / ((math.factorial(i)+1)**2)
                theta = random.random() * 2 * math.pi
                res[i,0,k,l] += radius*math.cos(theta)
                res[i,1,k,l] += radius*math.sin(theta)
    return res

def multi_factorial_decaying_random_initM(shape):
    res = []
    for i in range(shape[0]):
        if len(shape[1:]) == 4:
            res.append(factorial_decaying_random_initM(shape[1:]))
        else:
            res.append(multi_factorial_decaying_random_initM(shape[1:]))
    return np.array(res)


# This layer represets a power series
# A_0*I + A_1*X + A_2*X^2 + ... + A_n*X^n
# Where X is a complex input matrix, and the coefficients A_n are complex matrices.
# The optimized weights of this layer are the coefficients
class MatrixMPowerSeriesLayer(Layer):
    def __init__(self, degree, **kwrags):
        assert degree > 1
        self.degree = degree
        super().__init__(**kwrags)


    def build(self, input_shape):
        self.lcoefficients = self.add_weight(name='lrcoefficients',
                                            shape=(self.degree,*input_shape[1:]),
                                            initializer=factorial_decaying_random_initM,
                                            trainable=True)
        self.coefficients = ct.ComplexTensor(self.lcoefficients)
        self.unit = K.eye(input_shape[2])
        super().build(input_shape)

    def call(self, x):

        coefficients = ct.ComplexTensor(self.lcoefficients)

        x = ct.ComplexTensor(x)

        tmp = ct.ComplexTensor.unit_like(x)

        res = ct.compEinsum('jk,ikl->ijl', self.coefficients[0], tmp)
        
        for i in range(1, self.degree):
            tmp = ct.compEinsum('ijk,ikl->ijl', tmp, x)

            # Multiply by left coefficient
            res += ct.compEinsum('jk,ikl->ijl', self.coefficients[i], tmp)

        res = tf.stack([res.real, res.imag], axis=1)
        return res

    def compute_output_shape(self, input_shape):
        return input_shape
MMPS = MatrixMPowerSeriesLayer


# This layer represets a power series
# A_0*I*B_0 + A_1*X*B_1 + A_2*X^2*B_2 + ... + A_n*X^n*B_n
# Where X is a complex input matrix, and the coefficients A_n, B_n are complex matrices.
# The optimized weights of this layer are the coefficients
class MatrixM2PowerSeriesLayer(Layer):
    def __init__(self, degree, **kwrags):
        assert degree > 1
        self.degree = degree
        super().__init__(**kwrags)


    def build(self, input_shape):
        self.lrcoefficients = self.add_weight(name='lrcoefficients',
                                            shape=(self.degree,2,*input_shape[1:]),
                                            initializer=multi_factorial_decaying_random_initM,
                                            trainable=True)
        self.lcoefficients = ct.ComplexTensor(self.lrcoefficients[:,0])
        self.rcoefficients = ct.ComplexTensor(self.lrcoefficients[:,1])
        super().build(input_shape)

    def call(self, x):

        x = ct.ComplexTensor(x)

        tmp = ct.ComplexTensor.unit_like(x)

        # The unit matrix is "transperent" in matrix multiplication, therefore - there
        # is no need for both left and right coefficients
        res = ct.compEinsum('ijk,kl->ijl', tmp, self.rcoefficients[0])

        for i in range(1, self.degree):
            tmp = ct.compEinsum('ijk,ikl->ijl', tmp, x)

            # Multiply by right coefficient
            # Temporary results of right multiplication only to keep line degree managable
            rmul = ct.compEinsum('ijk,kl->ijl', tmp, self.rcoefficients[i])
            # Multiply by left coefficient and add to result
            res += ct.compEinsum('jk,ikl->ijl', self.lcoefficients[i], rmul)

        res = tf.stack([res.real, res.imag], axis=1)
        return res

    def compute_output_shape(self, input_shape):
        return input_shape
MM2PS = MatrixM2PowerSeriesLayer

def multi_factorial_decaying_random_init(shape):
    res = []
    for i in range(shape[0]):
        if len(shape[1:]) == 2:
            res.append(factorial_decaying_random_init(shape[1:]))
        else:
            res.append(multi_factorial_decaying_random_init(shape[1:]))
    return np.array(res)

# This is the same as MatrixPowerSeriesLayer, only for multiple channels of input and output.
# Input shape: [?(batch_size), k(input channels), 2(0=real\1=imag), n, n]
# Output shape: [?(batch_size), j(output channel), k(input channels), 2(0=real\1=imag), n, n]
# Calculates the same computations for every input channel in parallel
class MultichannelMatrixPowerSeriesLayer(Layer):
    def __init__(self, degree, out_channels, **kwrags):
        assert degree > 1
        self.degree = degree
        self.out_channels = out_channels
        super().__init__(**kwrags)


    def build(self, input_shape):
        self.coefficients = self.add_weight(name='coefficients',
                                            shape=(self.degree,self.out_channels,2),
                                            initializer=multi_factorial_decaying_random_init,
                                            trainable=True)
        self.coefficients = ct.ComplexTensor(self.coefficients, split_axis=-1)
        # self.coefficients is of shape [o,j]
        super().build(input_shape)

    def call(self, x):
        # convention:
        # i is batch size
        # j is output channels
        # k is input channels
        # l is 0=real\1=complex
        # m,n are elements of the input matrix
        # o - degree of polynomial

        x = ct.ComplexTensor(x)
        # x is a tensor of dimension [i,k,m,n]

        # tmp is used as the matrix raised to the n^th power
        tmp = ct.ComplexTensor.unit_like(x)

        # "tf.ones" is needed to raise dimension, this cannot be broadcasted later on
        res = ct.compEinsum('j,ikmn->ijkmn', self.coefficients[0], tmp)
        
        for o in range(1, self.degree):
            tmp = ct.compEinsum('ikmt,iktn->ikmn', tmp, x)

            # Update the result with the current element of the power series
            res += ct.compEinsum('j,ikmn->ijkmn', self.coefficients[o], tmp)

        # Unite real and complex parts
        res = tf.stack([res.real, res.imag], axis=3)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_channels, *input_shape[1:])
MchMPS = MultichannelMatrixPowerSeriesLayer

# This is the same as MatrixMPowerSeriesLayer, only for multiple channels of input and output
class MultichannelMatrixMPowerSeriesLayer(Layer):
    def __init__(self, degree, out_channels, **kwrags):
        assert degree > 1
        self.degree = degree
        self.out_channels = out_channels
        super().__init__(**kwrags)


    def build(self, input_shape):
        self.lcoefficients = self.add_weight(name='coefficients',
                                            shape=(self.degree,self.out_channels,2,*input_shape[-2:]),
                                            initializer=multi_factorial_decaying_random_initM,
                                            trainable=True)
        self.lcoefficients = ct.ComplexTensor(self.lcoefficients)
        self.unit = K.eye(input_shape[-1])
        super().build(input_shape)

    def call(self, x):
        # convention:
        # i is batch size
        # j is output channels
        # k is input channels
        # l is 0/1 - real/complex part
        # m,n,t are elements of matrices
        # o - degree of polynomial
        # This element ordering allows intuitive broadcasting

        x = ct.ComplexTensor(x)
        # [i,k,m,n]

        # tmp is used as the matrix raised to the n^th power
        tmp = unit_like(x)

        # "tf.ones" is needed to raise dimension, this cannot be broadcasted later on
        res = ct.compEinsum('jmt,iktn->ijkmn', self.lcoefficients, tmp)
        
        for o in range(1, self.degree):
            tmp = ct.compEinsum('ikmt,iktn->ikmn', tmp, x)

            # Update the result with the current element of the power series
            res += ct.compEinsum('ikmt,jtn->ijkmn', tmp, self.lcoefficients)

        # Unite real and complex parts
        res = tf.stack([res.real, res.imag], axis=3)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_channels, *input_shape[1:])
MchMMPS = MultichannelMatrixMPowerSeriesLayer

# This is the same as MatrixM2PowerSeriesLayer, only for multiple channels of input and output
class MultichannelMatrixM2PowerSeriesLayer(Layer):
    def __init__(self, degree, out_channels, **kwrags):
        assert degree > 1
        self.degree = degree
        self.out_channels = out_channels
        super().__init__(**kwrags)


    def build(self, input_shape):
        self.lrcoefficients = self.add_weight(name='coefficients',
                                            shape=(self.degree,self.out_channels,2,2,*input_shape[-2:]),
                                            initializer=multi_factorial_decaying_random_initM,
                                            trainable=True)
        self.lcoefficients = ct.ComplexTensor(self.lrcoefficients[:,:,0])
        self.rcoefficients = ct.ComplexTensor(self.lrcoefficients[:,:,1])
        self.unit = K.eye(input_shape[-1])
        super().build(input_shape)

    def call(self, x):
        # convention:
        # i is batch size
        # j is output channels
        # k is input channels
        # l is 0/1 - real/complex part
        # m,n,t are elements of matrices
        # o - degree of polynomial

        # This element ordering allows intuitive broadcasting

        x = ct.ComplexTensor(x)
        # x is a tensor of dimension [i,k,m,n]

        tmp = ct.ComplexTensor.unit_like(x)

        # On the 0th degree, there is no need for both left and right coefficient
        # so the left coefficient is discarded
        res = ct.compEinsum('ikmt,jtn', tmp, self.rcoefficients)
        
        for o in range(1, self.degree):
            tmp = ct.compEinsum('ikmt,iktn->ikmn', tmp, x)

            # Multiply by right coefficient
            # Temporary results of right multiplication only to keep line degree managable
            rmul = ct.compEinsum('ikmt,jtn->ijkmn', tmp, self.rcoefficients[o])
            res += ct.compEinsum('jmt,iktn->ijkmn', self.rcoefficients[o], rmul)

        # Unite real and complex parts
        res = tf.stack([res_real, res_imag], axis=3)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_channels, *input_shape[1:])
MchMM2PS = MultichannelMatrixM2PowerSeriesLayer
