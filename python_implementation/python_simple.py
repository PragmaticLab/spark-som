import sys
sys.path.insert(0, "../")
import random
import numpy as np
from sklearn.decomposition import PCA
from util import visualize_rgb

def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.
    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    return np.sqrt(np.dot(x, x.T))

class SOM:
	'''
	sigma(t) = sigma / (1 + 2 * t/T), controls the update gaussian
	'''
	def __init__(self, w, h, sigma=0.2, lr=0.1):
		self.w = w
		self.h = h
		self.neigx = np.arange(self.w)
		self.neigy = np.arange(self.h)
		self.sigma = sigma
		self.lr = lr
		self._decay_func = lambda x, t, max_iter: x/(1+float(t)/max_iter)
		
	'''
	methods:
	- random - random sample
	- PCA - pca grid
	'''
	def initialize(self, data):
		self.codebook = np.array([[np.zeros((3)) for i in range(self.w)] for j in range(self.h)])
		for j in range(self.h):
			for i in range(self.w):
				self.codebook[j][i] = random.choice(data)

	def train(self, data, iterations):
		random.shuffle(data)
		for t in range(iterations):
			diffbook = np.array([[np.zeros((3)) for i in range(self.w)] for j in range(self.h)])
			sigma = self._decay_func(self.sigma, t, iterations)
			lr = self._decay_func(self.lr, t, iterations)
			print t, sigma, lr
			for elem in data:
				(w_h, w_w) = self.winner(elem)
				g = self.gaussian((w_h, w_w), sigma) * lr
				it = np.nditer(g, flags=['multi_index'])
				while not it.finished:
					self.codebook[it.multi_index] += g[it.multi_index]*(elem - self.codebook[it.multi_index])
					self.codebook[it.multi_index] = self.codebook[it.multi_index] / fast_norm(self.codebook[it.multi_index])
					it.iternext()

			visualize_rgb(self.w, self.h, self.codebook, "iter_" + str(t))

	""" Computes the coordinates of the winning neuron for the sample x """
	def winner(self, x):
		min_dist = 999999
		min_coord = (0, 0)
		for j in range(self.h):
			for i in range(self.w):
				dist = np.linalg.norm(x-self.codebook[j][i])
				if dist <= min_dist:
					min_dist = dist
					min_coord = (j, i)
		return min_coord

	'''
	gaussian((2,2), 1)
	[[ 0.27992333  0.45123152  0.52907781  0.45123152]
	 [ 0.45123152  0.72737735  0.8528642   0.72737735]
	[ 0.52907781  0.8528642   1.          0.8528642 ]
	[ 0.45123152  0.72737735  0.8528642   0.72737735]]
	'''
	def gaussian(self, c, sigma):
		d = 2*np.pi*sigma*sigma
		ax = np.exp(-np.power(self.neigx-c[0], 2)/d)
		ay = np.exp(-np.power(self.neigy-c[1], 2)/d)
		return np.outer(ax, ay)


rgb = np.load("../data/generated_rgb.np")

s = SOM(6, 6, sigma=0.3, lr=0.5)
s.initialize(rgb)
visualize_rgb(s.w, s.h, s.codebook, "init")
s.train(rgb, 10)
visualize_rgb(s.w, s.h, s.codebook, "result")
