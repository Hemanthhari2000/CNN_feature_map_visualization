# Importing Libraries
import numpy as np 
from PIL import Image, ImageOps
import matplotlib.pyplot as plt 
from scipy import signal
from random import randint

SIZE = 64, 64
IMG_PATH = './imgs/bird1.jpg'

def resize_image(image):
	image = ImageOps.invert(image)
	image = image.resize(SIZE, Image.ANTIALIAS)
	return image

def load_image():
	image = Image.open(IMG_PATH).convert('L')
	# image = resize_image(image) # Resizing Image (optional tho...)
	image = np.asarray(image)
	image = image / 255.0
	return image

def get_kernel_array():
	kernel = []
	for _ in range(9):
		kernel.append(randint(-4,4))
	kernel = np.reshape(kernel, (3,3))
	return kernel

def show_kernel_and_image(kernel, image):
	print(f'Kernel:\n{kernel}')
	print(f'Image:\n{image}')


def show_convolved_image():
	image = load_image()
	kernel = get_kernel_array()
	show_kernel_and_image(kernel, image) # Kernel selected (Optional)
	grad = signal.convolve2d(image, kernel, mode='full')
	plt.imshow(grad, cmap=plt.cm.gray)
	plt.show()

def main():
	show_convolved_image()

if __name__ == '__main__':
	main()
