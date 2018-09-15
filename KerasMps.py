
from keras.engine.topology import Layer
from keras import backend as K
import tensorflow as tf
import numpy as np
import random
import math

def factorial_decaying_random_init(shape):
    reals = []
    imags = []
    for i in range(shape[0]):
        radius = random.random() / math.factorial(i)
        theta = random.random() * 2 * math.pi
        reals.append(radius*math.cos(theta))
        imags.append(radius*math.sin(theta))
    return np.array([reals, imags])


class MatrixPowerSeriesLayer(Layer):
    def __init__(self, length, **kwrags):
        assert length > 0
        self.length = length
        super().__init__(**kwrags)


    def build(self, input_shape):
        self.coefficients = self.add_weight(name='coefficients',
                                            shape=(self.length,2),
                                            initializer=factorial_decaying_random_init,
                                            trainable=True)
        self.unit = K.eye(input_shape[2])
        super().build(input_shape)

    def call(self, x):

        x_real = x[:,0]
        x_imag = x[:,1]

        tmp_real = x_real
        tmp_imag = x_imag

        coff_real = self.coefficients[0,:]
        coff_imag = self.coefficients[1,:]

        res_real = self.unit * coff_real[0] + tmp_real * coff_real[1] - tmp_imag * coff_imag[1]
        res_imag = self.unit * coff_imag[0] + tmp_real * coff_imag[1] + tmp_imag * coff_real[1]
        
        for i in range(2, self.length):
            new_tmp_real = tf.einsum('ijk,ikl->ijl', tmp_real, x_real) - tf.einsum('ijk,ikl->ijl', tmp_imag, x_imag)
            tmp_imag = tf.einsum('ijk,ikl->ijl', tmp_real, x_imag) + tf.einsum('ijk,ikl->ijl', tmp_imag, x_real)
            tmp_real = new_tmp_real

            res_imag += tmp_imag * coff_imag[i]
            res_real += tmp_real * coff_real[i]

        res = tf.stack([res_real, res_imag], axis=1)
        return res

    def compute_output_shape(self, input_shape):
        return input_shape

