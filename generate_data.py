''' mkdir data; python generate_data.py 10000 '''
import sys
import random
import numpy as np

r = lambda: random.randint(0,255)
# r = lambda: random.choice([0, 255])
count = int(sys.argv[1])


data = []
for i in range(count):
	data += [[r(), r(), r()]]
data = np.array(data).astype(np.float32)

f = open('data/generated_rgb.np', 'wb')
np.save(f, data)
np.savetxt("data/generated_rgb.csv", data, delimiter=",")
