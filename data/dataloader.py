import os
import numpy as np
from PIL import Image
import numpy as np

data_path = os.path.join(os.path.dirname(__file__),"images/")

def find_data_size():
	return len(os.listdir(data_path)) - 1 #data folder should always contain a number of images and a singe 'readme' file as a placeholder

data_size = find_data_size()

def get_img(idx):
	with Image.open(data_path + f"{idx}.png") as im:
		return (np.array(im).astype('float32') - 127.5) / 127.5

def get_batch(start : int, size : int):
	batch = []
	for i in range(start, start + size):
		batch.append(get_img(i))
	return np.array(batch)

def get_random_batch(size : int):
	batch = []
	for i in range(size):
		idx = np.random.randint(0, data_size)
		batch.append(get_img(idx))
	return np.array(batch)