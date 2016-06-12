''' mkdir data; python generate_data.py 10000 '''
import sys
import random
r = lambda: random.randint(0,255)

count = int(sys.argv[1])

f = open('data/generated_rgb.txt', 'wb')
for i in range(count):
	f.write(",".join(str(item) for item in [r(), r(), r()]))
	if i != count - 1:
		f.write("\n")
