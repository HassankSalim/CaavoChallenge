import tensorflow as tf
from input_pipeline import train_image_batch, train_label_batch, test_image_batch, test_label_batch, num_iter, BATCH_SIZE
import numpy as np
from opencv_utils import showImg
from sklearn.preprocessing import OneHotEncoder


ohe = OneHotEncoder()
ohe.fit(np.arange(15).reshape(-1, 1))

learning_rate = 0.001
epcohes = 10
batch_size = 128
display_step = 10

# def next_batch():

#     sess = tf.Session()
#     # initialize the variables
#     sess.run(tf.global_variables_initializer())

#     # initialize the queue threads to start to shovel data
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord, sess = sess)

#     print ("from the train set:")

#     for i in range(num_iter):
#         data, lab = sess.run([train_image_batch, train_label_batch])
#         data = data.reshape(BATCH_SIZE, 162, 209, 3).astype(np.float32) / 255
#         lab = ohe.transform(lab.reshape(-1, 1)).toarray()
#         showImg(data[0])
#         break
#     # stop our queue threads and properly close the session
    # coord.request_stop()
    # coord.join(threads)
    # sess.close()

# next_batch()

n_input = [None, 162, 209, 3]
n_classes = 15
dropout = 0.75

x = tf.placeholder(tf.float32, n_input)
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 64])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 256, 512])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 512, 512])),
    'wc6': tf.Variable(tf.random_normal([3, 3, 512, 512])),
    'wd1': tf.Variable(tf.random_normal([6*7*512, 4096])),
    'out': tf.Variable(tf.random_normal([4096, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bc4': tf.Variable(tf.random_normal([512])),
    'bc5': tf.Variable(tf.random_normal([512])),
    'bc6': tf.Variable(tf.random_normal([512])),
    'bd1': tf.Variable(tf.random_normal([4096])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def conv2d(x, W, b, strides=1):
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

def maxpool2d(x, k=2):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net_vgg7(x, weights, biases, dropout):
    
        x = tf.reshape(x, shape=[-1, 162, 209, 3])
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        conv1 = maxpool2d(conv1, 2)

        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = maxpool2d(conv2, 2)
        
        conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
        conv3 = maxpool2d(conv3, 2)
        
        conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
        conv4 = maxpool2d(conv4, 2)
        
        conv5 = conv2d(conv4, weights['wc5'], biases['bc5'])
        conv5 = maxpool2d(conv5, 2)
        
        conv6 = conv2d(conv5, weights['wc6'], biases['bc6'])
        conv6 = maxpool2d(conv6, 1)
        
        fc1 = tf.reshape(conv6, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        
        fc1 = tf.nn.dropout(fc1, dropout)
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        
        return out

pred = conv_net_vgg7(x, weights, biases, keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord, sess = sess)

	for i in range(epcohes):

		for j in range(num_iter):
			data, lab = sess.run([train_image_batch, train_label_batch])
			data = data.reshape(BATCH_SIZE, 162, 209, 3).astype(np.float32) / 255
			lab = ohe.transform(lab.reshape(-1, 1)).toarray()

			sess.run(optimizer, feed_dict={x: data, y: lab, keep_prob: dropout})
			print('single iter')
			
		loss, acc = sess.run([cost, accuracy], feed_dict={ x: data, y: lab, keep_prob: 1. })
		
		print("Epcohe " + i + ", Minibatch Loss= {:.6f}".format(loss) + ", Training Accuracy = {:.5f}".format(acc))

		print("Optimization Finished!")

	coord.request_stop()
	coord.join(threads)
	sess.close()