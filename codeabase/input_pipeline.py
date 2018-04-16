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


all_filenames, all_labels = read_filename_labels_from_csv('../dataset/best_dataset.csv')

total_size = len(all_labels)
test_set_size = int(0.2 * total_size)
train_set_size = total_size - test_set_size
num_iter = int(train_set_size / BATCH_SIZE)


all_images = ops.convert_to_tensor(all_filenames, dtype=dtypes.string)
all_labels = ops.convert_to_tensor(all_labels, dtype=dtypes.int32)

partitions = [0] * len(all_filenames)
partitions[:test_set_size] = [1] * test_set_size

train_images, test_images = tf.dynamic_partition(all_images, partitions, 2)
train_labels, test_labels = tf.dynamic_partition(all_labels, partitions, 2)

train_input_queue = tf.train.slice_input_producer(
                                    [train_images, train_labels],
                                    shuffle=True)

test_input_queue = tf.train.slice_input_producer(
                                    [test_images, test_labels],
                                    shuffle=True)

file_content = tf.read_file(train_input_queue[0])
train_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
train_label = train_input_queue[1]


file_content = tf.read_file(test_input_queue[0])
test_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
test_label = test_input_queue[1]

train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])


train_image_batch, train_label_batch = tf.train.batch(
                                    [train_image, train_label],
                                    batch_size=BATCH_SIZE
                                    )

test_image_batch, test_label_batch = tf.train.batch(
                                    [test_image, test_label],
                                    batch_size=BATCH_SIZE
                                    )

print("input pipeline ready")


