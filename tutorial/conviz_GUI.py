from __future__ import print_function
import numpy as np
import pygame
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tkinter import *
from PIL import Image
import pyscreenshot as ImageGrab
import numpy as np
import os

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

b1 = "up"
xold, yold = None, None
drawing_canvas, root, image_canvas, showing_result_canvas = None, None, None, None
label = None
targets = None

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
sess = tf.Session()
saver.restore(sess, "./bin/conv_mnist.ckpt")

def main():

    global drawing_canvas, root, image_canvas, showing_result_canvas, result_text
    root = Tk()
    root.title("Digits Recognition")  # set title
    root.resizable(0, 0)
    drawing_canvas = Canvas(root, width=28 * 10, height=28 * 10, background='white')
    drawing_canvas.grid(row=0, column=0, padx=5, sticky='nswe')
    drawing_canvas.bind("<Motion>", motion)
    drawing_canvas.bind("<ButtonPress-1>", b1down)
    drawing_canvas.bind("<ButtonRelease-1>", b1up)
    drawing_canvas.bind("<Button-3>",
                        lambda x: drawing_canvas.delete('all'))  # clear canvas when user click right button
    drawing_canvas.create_rectangle((10, 10, 270, 270))
    image_canvas = Canvas(root, width=28 * 10, height=28 * 10, bg='blue')
    image_canvas.grid(row=0, column=1, padx=5, pady=5)
    image_canvas.config(highlightthickness=1)
    showing_result_canvas = Canvas(root, height=60, background='greenyellow')
    showing_result_canvas.create_text(20, 40, anchor=W, font=("Purisa", 20),
                                      text="Draw any digit you want")
    showing_result_canvas.grid(row=1, column=0, columnspan=2, sticky=E+W)

    root.mainloop()  # start the event loop


def b1down(event):
    global b1
    b1 = "down"  # you only want to draw when the button is down
    # because "Motion" events happen -all the time-


def b1up(event):
    global b1, xold, yold, drawing_canvas, root, image_canvas
    b1 = "up"
    xold = None  # reset the line when you let go of the button
    yold = None
    # save fullsize image
    if not os.path.isdir('./tmp'):
        os.mkdir('./tmp')
    ImageGrab.grab().crop((root.winfo_x()+20, root.winfo_y() + 30, root.winfo_x()+250, root.winfo_y()+260)).save("./tmp/save.png")
    img = Image.open("./tmp/save.png")
    img_array = np.asarray(img)
    img = Image.fromarray(img_array, 'RGB').convert('L')

    # resize the image to fit the neural network
    img = img.resize((28, 28), Image.ANTIALIAS)  # best down-sizing filter
    data = (-(np.asfarray(img)-255.))/255.

    y_out = sess.run(predictions, feed_dict={x: [data.flatten()], keep_prob: dropout})[0]
    i = 0
    i_max = 0
    temp_prev = y_out[0]
    for temp in y_out:
        if temp > temp_prev:
            temp_prev = temp
            i_max = i
        i += 1
    print("The digit is a %s" % str(i_max))

    # display image after being processes
    img1 = img.resize((28 * 9 + 20, 28 * 9 + 20))
    img1.save("./tmp/save_scaled.gif")
    photo = PhotoImage(file = "./tmp/save_scaled.gif")
    image_canvas.create_image(5, 5, image = photo, anchor = 'nw')
    label = Label(image=photo)
    label.image = photo  # keep a reference!

def motion(event):
    if b1 == "down":
        global xold, yold, image, draw
        if xold is not None and yold is not None:
            # here's where you draw it. smooth. neat.
            event.widget.create_line(xold, yold, event.x, event.y, smooth=TRUE, width=20, capstyle=ROUND, joinstyle=ROUND)

        xold = event.x  # update x
        yold = event.y  # update y

if __name__ == "__main__":
    main()
