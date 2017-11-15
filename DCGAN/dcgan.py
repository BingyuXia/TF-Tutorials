import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import os
import scipy.misc
import scipy

#Dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

#Helper Functions
#Define a leaky relu activation
def lrelu(x, leak=0.2, name="lrelu"):
	with tf.variabel_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

#Save pictures
def inverse_transform(images):
	return (images + 1.) / 2.

def save_images(images, size, image_path):
	return isave(inverse_transform(images), size, image_path)

def isave(images, size, path):
	return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):    # images_shape (picture_number, height_pixel_number, weight_pixel_number)
	h, w = images.shape[1], images.shape[2]
	img = np.zeros(shape=(h * size[0], w * size[1]))

	for idx, image in enumerate(images):
		i = idx % size[1]
		j = idx // size[1]
		img[j*h : (j+1)*h, i*w : (i+1)*w] = image

	return img

#Generator Network
def generator(z):
	zP = slim.fully_connected(z, 4*4*256, normalizer_fn=slim.batch_norm, \
				activation_fn=tf.nn.relu, 
				weights_initializer=initializer, weights_regularizer=regularize)
	zCon = tf.reshape(zP, [-1, 4, 4, 256])

	padding = "SAME"
	with slim.arg_scope([slim.conv2d_transpose], padding=padding, activation_fn=tf.nn.relu,
				weights_initializer=initializer, weights_regularizer=regularize)
		gen1 = slim.conv2d_transpose(zCon, 64, [5, 5], 2, scope="g_conv1")
		gen2 = slim.conv2d_transpose(gen1, 32, [5, 5], 2, scope="g_conv2")
		gen3 = slim.conv2d_transpose(gen2, 16, [5, 5], 2, scope="g_conv3")

	gout = slim.conv2d_transpose(gen3, 1, [32, 32], padding=padding, biases_initializer=None,
				activation_fn=tf.nn.tanh, weights_initializer=initializer, weights_regularizer=regularize, 
				scope="g_out")

#Discriminator Network
#Input size: 32 * 32
def 



#Main
initializer = tf.truncated_normal_initializer(stddev=0.02)
regularize = slim.l2_regularizer(0.0005)