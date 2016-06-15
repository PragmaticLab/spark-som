import random
import numpy as np
from util import visualize_rgb
from minisom import MiniSom    

rgb = np.load("../data/generated_rgb.np")

w, h = 6, 6

som = MiniSom(w, h, 3, sigma=0.3, learning_rate=0.5)

visualize_rgb(w, h, som.weights, filename="minisom_init")

print "Training..."

som.train_random(rgb, 100)

print "...ready!"

visualize_rgb(w, h, som.weights, filename="minisom_result")
