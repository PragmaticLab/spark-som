"""
spark-submit --conf spark.driver.maxResultSize=1g --conf spark.yarn.executor.memoryOverhead=800 --num-executors 6 --executor-cores 1 --driver-memory 500m  --executor-memory 500m spark_som.py 
"""
import sys
sys.path.insert(0, "../")
import random
import numpy as np
from sklearn.decomposition import PCA
from util import visualize_rgb

def fast_norm(x):
	return np.sqrt(np.dot(x, x.T))
def gaussian(w, h, c, sigma):
	d = 2*np.pi*sigma*sigma
	neigx = np.arange(w)
	neigy = np.arange(h)
	ax = np.exp(-np.power(neigx-c[0], 2)/d)
	ay = np.exp(-np.power(neigy-c[1], 2)/d)
	return np.outer(ax, ay)
def winner(x, codebook, w, h):
	activation_map = np.zeros((h, w))
	s = np.subtract(x, codebook)
	it = np.nditer(activation_map, flags=['multi_index'])
	while not it.finished:
		activation_map[it.multi_index] = fast_norm(s[it.multi_index])  
		it.iternext()
	return np.unravel_index(activation_map.argmin(), activation_map.shape)

class SOM:
	def __init__(self, w, h, sigma=0.2, lr=0.1):
		self.w = w
		self.h = h
		self.neigx = np.arange(self.w)
		self.neigy = np.arange(self.h)
		self.sigma = sigma
		self.lr = lr
		# self._decay_func = lambda x, t, max_iter: x - (x - 0.000001) * float(t) / max_iter
		self._decay_func = lambda x, t, max_iter: x/(1+float(t)/max_iter)
		
	def initialize(self, data):
		self.codebook = np.array([[np.zeros((3)) for i in range(self.w)] for j in range(self.h)])
		for j in range(self.h):
			for i in range(self.w):
				self.codebook[j][i] = random.choice(data)

	def quantization_error(self, data):
		error = 0
		for x in data:
			error += fast_norm(x-self.codebook[winner(x, self.codebook, self.w, self.h)])
		return error/len(data)

	def train(self, data, iterations, partitions=12):
		from pyspark import SparkContext
		sc = SparkContext()
		dataRDD = sc.parallelize(data).cache()
		for t in range(iterations):
			sigma = self._decay_func(self.sigma, t, iterations)
			lr = self._decay_func(self.lr, t, iterations)
			codebookBC = sc.broadcast(self.codebook)
			randomizedRDD = dataRDD.repartition(partitions)
			print "iter: %d, sigma: %.2f, lr: %.2f, error: %.4f" % (t, sigma, lr, self.quantization_error(randomizedRDD.collect()))
			def train_partition(partition_data):
				localCodebook = codebookBC.value
				for elem in partition_data:
					(w_h, w_w) = winner(elem, localCodebook, self.w, self.h)
					g = gaussian(self.w, self.h, (w_h, w_w), sigma) * lr
					it = np.nditer(g, flags=['multi_index'])
					while not it.finished:
						localCodebook[it.multi_index] += g[it.multi_index]*(elem - localCodebook[it.multi_index])
						it.iternext()
				return [localCodebook]
			resultCodebookRDD = randomizedRDD.mapPartitions(train_partition)
			sumCodebook = resultCodebookRDD.reduce(lambda a, b: a + b)
			newCodebook = sumCodebook / float(partitions)
			self.codebook = newCodebook


rgb = np.load("../data/generated_rgb.np")
rgb /= 255.0

s = SOM(6, 6, sigma=0.5, lr=0.3)

s.initialize(rgb)

visualize_rgb(s.w, s.h, s.codebook, "spark_init")
s.train(rgb, 20)
visualize_rgb(s.w, s.h, s.codebook, "spark_result")
