
import tensorflow as tf

class ComplexTensor:
    def __init__(self, tensor, split_axis=-3):
        if isinstance(tensor, list):
            self.real = tensor[0]
            self.imag = tensor[1]
        else:
            # Assume this is a tensorflow tensor -
            # there are too many types of those to check manually
            self.real, self.imag = tf.split(tensor, [1,1], axis=split_axis)
            self.real = tf.squeeze(self.real, axis=split_axis)
            self.imag = tf.squeeze(self.imag, axis=split_axis)
        self.dim = len(self.real.shape)


    @classmethod
    def unit_like(cls, compTens, merge_axis=-3):
        imag = tf.zeros_like(compTens.imag)
        real = tf.zeros_like(compTens.real)
        real += tf.eye(int(compTens.real.shape[-1]))
        return cls([real, imag])

    def __getitem__(self, val):
        real = self.real[val]
        imag = self.imag[val]
        return ComplexTensor([real, imag])

    def __iadd__(self, other):
        self.real += other.real
        self.imag += other.imag
        return self

    def __add__(self, other):
        real = self.real+other.real
        imag = self.imag+other.imag
        return ComplexTensor([real, imag])

    def __imul__(self, other):
        assert other.dim == 0
        tmp = self.real*other.real - self.imag*other.imag
        self.imag = self.real*other.imag + self.imag*other.real
        self.real = tmp
        return self

    def __mul__(self, other):
        assert other.dim == 0
        real = self.real*other.real - self.imag*other.imag
        imag = self.real*other.imag + self.imag*other.real
        return ComplexTensor([real, imag])

def compEinsum(form, a, b, split_axis=-3):
    real = tf.einsum(form, a.real, b.real) - tf.einsum(form, a.imag, b.imag)
    imag = tf.einsum(form, a.real, b.imag) + tf.einsum(form, a.imag, b.real)
    return ComplexTensor([real, imag])

