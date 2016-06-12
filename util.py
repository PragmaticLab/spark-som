import numpy as np 
from pylab import imread,imshow,figure,show,subplot,title,savefig


def visualize_rgb(w, h, codebook, filename="test"):
	figure(1)
	subplot(221)
	title('rgb visualization')
	imshow(codebook, interpolation="nearest")
	# imshow(codebook)
	# show()
	savefig("results/" + filename + ".png")
	