import sys
sys.path.insert(0, "../")
import random
import numpy as np
from sklearn.decomposition import PCA
from util import visualize_rgb

def fast_norm(x):
    return np.sqrt(np.dot(x, x.T))

class SOM:
	def __init__(self, w, h, sigma=0.2, lr=0.1):
		self.w = w
		self.h = h
		self.neigx = np.arange(self.w)
		self.neigy = np.arange(self.h)
		self.sigma = sigma
		self.lr = lr
		self._decay_func = lambda x, t, max_iter: x/(1+float(t)/max_iter)
		
	def initialize(self, data):
		self.codebook = np.array([[np.zeros((3)) for i in range(self.w)] for j in range(self.h)])
		for j in range(self.h):
			for i in range(self.w):
				self.codebook[j][i] = random.choice(data)

	def train(self, data, iterations):
		random.shuffle(data)
		for t in range(iterations):
			sigma = self._decay_func(self.sigma, t, iterations)
			lr = self._decay_func(self.lr, t, iterations)
			print t, sigma, lr
			for elem in data:
				(w_h, w_w) = self.winner(elem)
				g = self.gaussian((w_h, w_w), sigma) * lr
				it = np.nditer(g, flags=['multi_index'])
				while not it.finished:
					# print elem - self.codebook[it.multi_index]
					self.codebook[it.multi_index] += g[it.multi_index]*(elem - self.codebook[it.multi_index])
					# print g[it.multi_index]*(elem - self.codebook[it.multi_index])
					self.codebook[it.multi_index] = self.codebook[it.multi_index] / fast_norm(self.codebook[it.multi_index])
					it.iternext()

			visualize_rgb(self.w, self.h, self.codebook, "iter_" + str(t))

	def _activate(self, x):
		self.activation_map = np.zeros((self.h,self.w))
		s = np.subtract(x, self.codebook)
		it = np.nditer(self.activation_map, flags=['multi_index'])
		while not it.finished:
			self.activation_map[it.multi_index] = fast_norm(s[it.multi_index])  
			it.iternext()

	def winner(self, x):
		self._activate(x)
		return np.unravel_index(self.activation_map.argmin(), self.activation_map.shape)

	def gaussian(self, c, sigma):
		d = 2*np.pi*sigma*sigma
		ax = np.exp(-np.power(self.neigx-c[0], 2)/d)
		ay = np.exp(-np.power(self.neigy-c[1], 2)/d)
		return np.outer(ax, ay)


rgb = np.load("../data/generated_rgb.np")
# rgb /= 255.0

s = SOM(6, 6, sigma=0.3, lr=0.3)

s.initialize(rgb)

visualize_rgb(s.w, s.h, s.codebook, "init")
s.train(rgb, 10)
visualize_rgb(s.w, s.h, s.codebook, "result")
