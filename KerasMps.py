
from keras.engine.topology import Layer
from keras import backend as K
import numpy as np
import random
import math

def factorial_decaying_random_init(shape):
	coefficients = []
	for i in range(shape[0]):
		coefficients.append(random.random() + random.random()*(1j) / math.factorial(i))
	return np.array(coefficients, dtype=np.complex_)

class MatrixPowerSeriesLayer(Layer):
	def __init__(self, length, **kwrags):
		self.length = length
		super().__init__(**kwrags)

	def build(self, input_shape):
		self.coefficients = self.add_weight(name='coefficients',
											shape=(self.length,),
											initializer=factorial_decaying_random_init,
											trainable=True)
		self.unit = K.eye(input_shape[1])
		super().build(input_shape)

	def call(self, x):
		tmp = K.variable(self.unit)
		print(x.shape)
		res = self.coefficients[0]*tmp
		for i in range(1,self.length):
			print(tmp.shape)
			tmp = K.dot(tmp, x)
			print(tmp.shape)
			res += self.coefficients[i]*tmp
		print(res.shape)
		return res

	def compute_output_shape(self, input_shape):
		return input_shape

