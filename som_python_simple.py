import random
import numpy as np
from sklearn.decomposition import PCA
from util import visualize_rgb

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
		self.sigma_decay_func = lambda x, curr_iter, max_iter: x / (1 + 2 * float(curr_iter) / max_iter)
		self.lr_decay_func = lambda x, curr_iter, max_iter: x - (float(curr_iter) / max_iter) * x + 0.00000001
	
	'''
	methods:
	- random - random sample
	- PCA - pca grid
	'''
	def initialize(self, data, method="random"):
		self.codebook = np.array([[np.zeros((3)) for i in range(self.w)] for j in range(self.h)])
		if method == "random":
			for j in range(self.h):
				for i in range(self.w):
					self.codebook[j][i] = random.choice(data)
		elif method == "PCA":
			# note this is very underdeveloped, ideally should not just take 1st principal component
			pca = PCA(n_components=1)
			transformed = pca.fit_transform(data)
			dim = self.w * self.h
			ranges = range(np.min(transformed), np.max(transformed), int((np.max(transformed) - np.min(transformed)) / float(dim)))
			for i, (trans_min, trans_max) in enumerate(zip(ranges[:len(ranges)-1], ranges[1:])):
				h = i / self.w
				w = i % self.w
				while True:
					index = int(random.random() * transformed.shape[0])
					if transformed[index] > trans_min and transformed[index] < trans_max:
						self.codebook[h][w] = data[index]
						break


	def train(self, data, iterations):
		random.shuffle(data)
		for t in range(iterations):
			diffbook = np.array([[np.zeros((3)) for i in range(self.w)] for j in range(self.h)])
			sigma = self.sigma_decay_func(self.sigma, t, iterations)
			lr = self.lr_decay_func(self.lr, t, iterations)
			print t, sigma, lr
			# print self.gaussian((2,2), sigma)
			for batch in np.split(data, data.shape[0] / 1000):
				for elem in batch:
					(w_h, w_w) = self.winner(elem)
					winner = self.codebook[w_h][w_w]
					diff = elem - winner
					g = self.gaussian((w_h, w_w), sigma)
					for j in range(self.h):
						for i in range(self.w):
							diffbook[j][i] += g[j][i] * diff
				diffbook = diffbook / float(batch.shape[0])
				# print diffbook * lr
				self.codebook += diffbook * lr
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


rgb = np.load("data/generated_rgb.np")

s = SOM(4, 4, sigma=1, lr=0.5)
# s.initialize(rgb, "PCA")
s.initialize(rgb, "random")
visualize_rgb(s.w, s.h, s.codebook, "init")
# print s.gaussian((2, 2), 0.1)
s.train(rgb, 20)
visualize_rgb(s.w, s.h, s.codebook, "result")
