
from keras.engine.topology import Layer
from keras import backend as K
import tensorflow as tf
import numpy as np
import random
import math

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
# The optimized weights of this layer are the coefficients
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
        self.unit = K.eye(input_shape[2])
        super().build(input_shape)

    def call(self, x):
        # Extract the real and imaginary parts of the input matrix
        x_real = x[:,0]
        x_imag = x[:,1]

        # tmp is used as the matrix raised to the n^th power
        tmp_real = tf.zeros_like(x_real) + self.unit
        tmp_imag = tf.zeros_like(x_imag)

        # Find real and imaginary coefficients of the power series
        coff_real = self.coefficients[:,0]
        coff_imag = self.coefficients[:,1]

        # Initialize with the zeorth power of the series
        res_real = self.unit * coff_real[0]
        res_imag = self.unit * coff_imag[0]
        
        for i in range(1, self.degree):
            # Calculate a raise by one of the current power of the matrix
            # Remainder: (a+bj)(c+dj)=(ac-bd)+(ad+bc)j, and the same 
            # formulas hold for matrices
            new_tmp_real = tf.einsum('ijk,ikl->ijl', tmp_real, x_real) - tf.einsum('ijk,ikl->ijl', tmp_imag, x_imag)
            tmp_imag = tf.einsum('ijk,ikl->ijl', tmp_real, x_imag) + tf.einsum('ijk,ikl->ijl', tmp_imag, x_real)
            tmp_real = new_tmp_real

            # Update the result with the current element of the power series
            res_real += tmp_real * coff_real[i] - tmp_imag * coff_imag[i]
            res_imag += tmp_imag * coff_real[i] + tmp_real * coff_imag[i]

        # Unite real and complex parts
        res = tf.stack([res_real, res_imag], axis=1)
        return res

    def compute_output_shape(self, input_shape):
        return input_shape



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
        self.unit = K.eye(input_shape[2])
        super().build(input_shape)

    def call(self, x):

        x_real = x[:,0]
        x_imag = x[:,1]

        tmp_real = tf.zeros_like(x_real) + self.unit
        tmp_imag = tf.zeros_like(x_imag)

        # The coefficient matrices
        lcoff_real = self.lcoefficients[:,0]
        lcoff_imag = self.lcoefficients[:,1]

        res_real = tf.einsum('jk,kl->jl', lcoff_real[0], self.unit)
        res_imag = tf.einsum('jk,kl->jl', lcoff_imag[0], self.unit)
        
        for i in range(1, self.degree):
            new_tmp_real = tf.einsum('ijk,ikl->ijl', tmp_real, x_real) - tf.einsum('ijk,ikl->ijl', tmp_imag, x_imag)
            tmp_imag = tf.einsum('ijk,ikl->ijl', tmp_real, x_imag) + tf.einsum('ijk,ikl->ijl', tmp_imag, x_real)
            tmp_real = new_tmp_real

            # Multiply by left coefficient
            res_real += tf.einsum('jk,ikl->ijl', lcoff_real[i], tmp_real) - tf.einsum('jk,ikl->ijl', lcoff_imag[i], tmp_imag)
            res_imag += tf.einsum('jk,ikl->ijl', lcoff_imag[i], tmp_real) + tf.einsum('jk,ikl->ijl', lcoff_real[i], tmp_imag)

        res = tf.stack([res_real, res_imag], axis=1)
        return res

    def compute_output_shape(self, input_shape):
        return input_shape

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
        self.lcoefficients = self.lrcoefficients[:,0]
        self.rcoefficients = self.lrcoefficients[:,1]
        self.unit = K.eye(input_shape[2])
        super().build(input_shape)

    def call(self, x):

        x_real = x[:,0]
        x_imag = x[:,1]

        tmp_real = tf.zeros_like(x_real) + self.unit
        tmp_imag = tf.zeros_like(x_imag)

        # The coefficient matrices
        rcoff_real = self.rcoefficients[:,0]
        rcoff_imag = self.rcoefficients[:,1]
        lcoff_real = self.lcoefficients[:,0]
        lcoff_imag = self.lcoefficients[:,1]

        # The unit matrix is "transperent" in matrix multiplication, therefore - there
        # is no need for both left and right coefficients
        res_real = tf.einsum('jk,kl->jl', self.unit, rcoff_real[0])
        res_imag = tf.einsum('jk,kl->jl', self.unit, rcoff_imag[0])
        
        for i in range(1, self.degree):
            new_tmp_real = tf.einsum('ijk,ikl->ijl', tmp_real, x_real) - tf.einsum('ijk,ikl->ijl', tmp_imag, x_imag)
            tmp_imag = tf.einsum('ijk,ikl->ijl', tmp_real, x_imag) + tf.einsum('ijk,ikl->ijl', tmp_imag, x_real)
            tmp_real = new_tmp_real

            # Multiply by right coefficient
            # Temporary results of right multiplication only to keep line degree managable
            rmul_real = tf.einsum('ijk,kl->ijl', tmp_real, rcoff_real[i]) - tf.einsum('ijk,kl->ijl', tmp_imag, rcoff_imag[i])
            rmul_imag = tf.einsum('ijk,kl->ijl', tmp_real, rcoff_imag[i]) + tf.einsum('ijk,kl->ijl', tmp_imag, rcoff_real[i])

            # Multiply by left coefficient
            res_real += tf.einsum('jk,ikl->ijl', lcoff_real[i], rmul_real) - tf.einsum('jk,ikl->ijl', lcoff_imag[i], rmul_imag)
            res_imag += tf.einsum('jk,ikl->ijl', lcoff_imag[i], rmul_real) + tf.einsum('jk,ikl->ijl', lcoff_real[i], rmul_imag)

        res = tf.stack([res_real, res_imag], axis=1)
        return res

    def compute_output_shape(self, input_shape):
        return input_shape


def multi_factorial_decaying_random_init(shape):
    res = []
    for i in range(shape[0]):
        if len(shape[1:]) == 2:
            res.append(factorial_decaying_random_init(shape[1:]))
        else:
            res.append(multi_factorial_decaying_random_init(shape[1:]))
    return np.array(res)

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
        self.unit = K.eye(input_shape[-1])
        super().build(input_shape)

    def call(self, x):
        # convention:
        # i is batch size
        # j is output channels
        # k is input channels
        # l is 0/1 - real/complex part
        # m,n are elements of the input matrix
        # o - degree of polynomial

        # This element ordering allows intuitive broadcasting
        # x is now a tensor of dimension [i,k,l,m,n]

        x_real = x[:,:,0]
        x_imag = x[:,:,1]
        # x_real/imag is a tensor of dimension [i,k,m,n]

        # tmp is used as the matrix raised to the n^th power
        tmp_real = tf.zeros_like(x_real) + self.unit
        tmp_imag = tf.zeros_like(x_imag)
        # tmp_real/imag is a tensor of dimension [i,k,m,n]

        coff_real = self.coefficients[:,:,0]
        coff_imag = self.coefficients[:,:,1]
        # coff_real/imag is now a matrix [o,j]

        res_real = tf.einsum('j,k,mn->jkmn', coff_real[0], tf.ones((x.shape[1],)), self.unit)
        res_imag = tf.einsum('j,k,mn->jkmn', coff_imag[0], tf.ones((x.shape[1],)), self.unit)

        
        for o in range(1, self.degree):
            new_tmp_real = tf.einsum('ikmn,iknl->ikml', tmp_real, x_real) - tf.einsum('ikmn,iknl->ikml', tmp_imag, x_imag)
            tmp_imag = tf.einsum('ikmn,iknl->ikml', tmp_real, x_imag) + tf.einsum('ikmn,iknl->ikml', tmp_imag, x_real)
            tmp_real = new_tmp_real

            # Update the result with the current element of the power series
            res_real += tf.einsum('ikmn,j->ijkmn', tmp_real, coff_real[o]) - tf.einsum('ikmn,j->ijkmn', tmp_imag, coff_imag[o])
            res_imag += tf.einsum('ikmn,j->ijkmn', tmp_imag, coff_real[o]) + tf.einsum('ikmn,j->ijkmn', tmp_real, coff_imag[o])

        # Unite real and complex parts
        res = tf.stack([res_real, res_imag], axis=3)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.out_channels, *input_shape[1:])
