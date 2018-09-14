
import numpy as np

class MatrixPowerSeries:
	def __init__(self, size = 0, series = None):
		self._size = size
		self._series = []
		if series is not None:
			self._series = series

	def copmute(self, M):
		# I belive topological properties of the matrix space takes care of
		# degenerate non-diagonizable cases together with the numerical
		# implementation of numpy. Better hope nobody sends us an hilbert
		# matrix!
		values, vectors = np.linalg.eig(M)

		# P is the transition to base matrix
		P = vectors
		Pi = inv(P)
		D = np.diag(values)

		_sum = self._series[0] * np.identity(M.shape[0], dtype=np.complex_)
		for i in range(1,self.size):
			_sum += self._series[i] * P.dot(D**i).dot(Pi)

		return _sum
