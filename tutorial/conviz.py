from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import errno
import shutil
from tensorflow.examples.tutorials.mnist import input_data

PLOT_DIR = './plots'

def get_grid_dim(x):
    """
    Transforms x into product of two integers
    :param x: int
    :return: two ints
    """
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]


def prime_powers(n):
    """
    Compute the factors of a positive integer
    Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    :param n: int
    :return: set
    """
    factors = set()
    for x in range(1, int(math.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)


def empty_dir(path):
    """
    Delete all files and folders in a directory
    :param path: string, path to directory
    :return: nothing
    """
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Warning: {}'.format(e))


def create_dir(path):
    """
    Creates a directory
    :param path: string
    :return: nothing
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


def prepare_dir(path, empty=False):
    """
    Creates a directory if it soes not exist
    :param path: string, path to desired directory
    :param empty: boolean, delete all directory content if it exists
    :return: nothing
    """
    if not os.path.exists(path):
        create_dir(path)

    if empty:
        empty_dir(path)


def plot_conv_weights(weights, name, channels_all=True):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_weights')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    prepare_dir(plot_dir, empty=True)

    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    num_filters = weights.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate channels
    for channel in channels:
        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = weights[:, :, channel, l]
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')


def plot_conv_output(conv_img, name):
    """
    Makes plots of results of performing convolution
    :param conv_img: numpy array of rank 4
    :param name: string, name of convolutional layer
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_output')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    prepare_dir(plot_dir, empty=True)

    w_min = np.min(conv_img)
    w_max = np.max(conv_img)

    # get number of convolutional filters
    num_filters = conv_img.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate filters
    for l, ax in enumerate(axes.flat):
        # get a single image
        img = conv_img[0, :, :,  l]
        # put it on the grid
        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
        # remove any labels from the axes
        ax.set_xticks([])
        ax.set_yticks([])
    # save figure
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')

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
learning_rate = 1e-4
training_epochs = 1000
batch_size = 50
display_step = 100
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.5  # Dropout, probability to keep units
features1 = 2
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
# y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# Define loss operation
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(predictions), reduction_indices=[1]))

# Define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# Define accuracy operation
correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 0
    # Keep training until reach max iterations
    while step < training_epochs:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y_: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.})
            print("Iteration " + str(step) + ", Training Accuracy= {:.5f}".format(train_accuracy), end='\n')
        step += 1

    # Calculate accuracy on the test set
    test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.})
    print("Test Accuracy:", test_accuracy, end='\n')

    # get weights of all convolutional layers
    # no need for feed dictionary here
    conv_weights = sess.run([tf.get_collection('conv_weights')])
    for i, c in enumerate(conv_weights[0]):
        plot_conv_weights(c, 'conv{}'.format(i))

    # get output of all convolutional layers
    # here we need to provide an input image
    conv_out = sess.run([tf.get_collection('conv_output')], feed_dict={x: mnist.test.images[:1]})
    for i, c in enumerate(conv_out[0]):
        plot_conv_output(c, 'conv{}'.format(i))


