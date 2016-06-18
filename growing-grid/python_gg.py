import numpy as np 
import sys
sys.path.insert(0, "../")
from util import visualize_rgb

class GrowingGrid:
	def __init__(self, sigma=0.2, lr=0.1):
		self.w = 2
		self.h = 2
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

rgb = np.load("../data/generated_rgb.np")

gg = GrowingGrid(sigma=0.5, lr=0.5)
gg.initialize(rgb)
