from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Weights initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.5  # Dropout, probability to keep units
features1 = 8
features2 = 2 * features1

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y_ = tf.placeholder(tf.float32, [None, n_classes])

# Reshape inputs
x_reshaped = tf.reshape(x, shape=[-1, 28, 28, 1])

# First convolutional layer
W_conv1 = weight_variable([5, 5, 1, features1])
tf.add_to_collection('conv_weights', W_conv1)
b_conv1 = bias_variable([features1])
h_conv1 = tf.nn.relu(conv2d(x_reshaped, W_conv1) + b_conv1)
tf.add_to_collection('conv_output', h_conv1)

# First max-pooling layer
h_pool1 = max_pool_2x2(h_conv1)

# Second convolutional layer
W_conv2 = weight_variable([5, 5, features1, features2])
tf.add_to_collection('conv_weights', W_conv2)
b_conv2 = bias_variable([features2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
tf.add_to_collection('conv_output', h_conv2)

# Second max-pooling layer
h_pool2 = max_pool_2x2(h_conv2)

# Densely connected layer
W_fc1 = weight_variable([7 * 7 * features2, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * features2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout (reduces overfitting)
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# Output
predictions = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Define accuracy operation
correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Add ops to save and restore all the variables
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:

    saver.restore(sess, "./bin/conv_mnist.ckpt")

    # Calculate accuracy on the test set
    test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.})
    print("Test Accuracy:", test_accuracy, end='\n')
