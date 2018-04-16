import tensorflow as tf
from glob import glob
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


NUM_CHANNELS = 3
IMAGE_HEIGHT = 162
IMAGE_WIDTH = 209
BATCH_SIZE = 256
ohe = OneHotEncoder()
ohe.fit(np.arange(15).reshape(-1, 1))


def encodeFromFilename(filename):
	label = filename.split('/')[3]
	return int(label)

def read_filename_labels_from_csv(csv_file):
	df = pd.read_csv(csv_file)
	paths = df['path']
	filenames = []
	labels = []
	for filename in paths:
		filenames.append(filename)
		labels.append(encodeFromFilename(filename))
	return filenames, labels


	