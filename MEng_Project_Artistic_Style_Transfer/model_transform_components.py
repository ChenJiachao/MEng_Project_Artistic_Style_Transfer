from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

# Import the normalization parameters
# Using the conditional instance normalization fn
import normalization as nl 

slim = tf.contrib.slim



# As we read from the paper, we freeze the normalization parameters
# therefore, in the transform function, we need to have normalaizer_params

# ATTENTION: THIS TRANSFORM FUNCTION NEVER BE USED IN THE BUILD MODEL
def transform( 	
				input_,
				normalizer_fn = nl.conditional_instance_norm,
				normalizer_params = None,
				reuse = False):

	# if we dont have normalization parameters
	# then, center: substract beta is true:
	# then, gamma: multiply gamma is true
	if normalizer_params is None:
		normalizer_params = {'center': True,
								'scale': True}

	with tf.variable_scope('transformer', reuse = reuse):
		
		with slim.arg_scope( [slim.conv2d],
				activation_fn = tf.nn.relu,
				normalizer_fn = normalizer_fn,
				normalizer_params = normalizer_params,
				weights_initializer = tf.random_normal_initializer(0.0, 0.01),
				biases_initializer = tf.constant_initializer(0.0)
									):
			

			# While the original code said it should be contract
			# However, I think it should say convolution
			with tf.variable_scope('contract'):
				h = conv2d(input_, 9, 1, 32, 'conv1')
				h = conv2d(h, 3, 2, 64, 'conv2')
				h = conv2d(h, 3, 2, 128, 'conv3')
			with tf.variable_scope('residual'):
				h = residual_block(h, 3, 'residual1')
				h = residual_block(h, 3, 'residual2')
				h = residual_block(h, 3, 'residual3')
				h = residual_block(h, 3, 'residual4')
				h = residual_block(h, 3, 'residual5')
			with tf.variable_scope('expand'):
				h = upsampling(h, 3, 2, 64, 'conv1')
				h = upsampling(h, 3, 2, 32, 'conv2')
				return upsampling(h, 9, 1, 3, 'conv3', activation_fn=tf.nn.sigmoid)



def conv2d( input_,
			kernel_size,
			stride,
			num_outputs,
			scope,
			activation_fn = tf.nn.relu
			):
	
	# WHY????????????????????????????????
	if kernel_size % 2 == 0:
		raise ValueError('kernel_size is expected to be odd.')

	padding = kernel_size // 2


	# About the padding part, it looks like a new strategy compared to the CV course

	'''
	t = tf.constant([[1, 2, 3], [4, 5, 6]])
	paddings = tf.constant([[1, 1,], [2, 2]])
	# 'constant_values' is 0.
	# rank of 't' is 2.
	tf.pad(t, paddings, "CONSTANT")  # [[0, 0, 0, 0, 0, 0, 0],
								 #  [0, 0, 1, 2, 3, 0, 0],
								 #  [0, 0, 4, 5, 6, 0, 0],
								 #  [0, 0, 0, 0, 0, 0, 0]]

	tf.pad(t, paddings, "REFLECT")  # [[6, 5, 4, 5, 6, 5, 4],
								#  [3, 2, 1, 2, 3, 2, 1],
								#  [6, 5, 4, 5, 6, 5, 4],
								#  [3, 2, 1, 2, 3, 2, 1]]

	tf.pad(t, paddings, "SYMMETRIC")  # [[2, 1, 1, 2, 3, 3, 2],
								  #  [2, 1, 1, 2, 3, 3, 2],
								  #  [5, 4, 4, 5, 6, 6, 5],
								  #  [5, 4, 4, 5, 6, 6, 5]]
	'''

	padded_input = tf.pad(	input_,					
				[[0,0], [padding,padding], [padding,padding], [0,0]],
				mode = "REFLECT"
							)
	return slim.conv2d(
	  padded_input,
	  padding='VALID',
	  kernel_size=kernel_size,
	  stride=stride,
	  num_outputs=num_outputs,
	  activation_fn=activation_fn,
	  scope=scope)


# 
def upsampling (input_,
		kernel_size,
		stride,
		num_outputs,
		scope,
		activation_fn = tf.nn.relu
					):

	
	if kernel_size % 2 == 0:
		raise ValueError('kernel_size is expected to be odd.')
	with tf.variable_scope(scope):
		shape = tf.shape(input_)
		# As explained earlier,
		# height, width corespond to the index 1, 2
		height = shape[1]
		width = shape [2]

		# This is the way to upsampling
		upsampled_input = tf.image.resize_nearest_neighbor( 
								input_,
								[stride * height,
								stride * width ]
								)

		return conv2d(	upsampled_input,
				kernel_size,
				1,
				num_outputs,
				'conv',
				activation_fn = activation_fn
						)

def residual_block(	input_,
			kernel_size,
			scope,
			activation_fn = tf.nn.relu):
	
	if kernel_size % 2 == 0:
		raise ValueError('kernel_size is expected to be odd.')
	with tf.variable_scope(scope):
		num_outputs = input_.get_shape()[-1].value
		h_1 = conv2d(input_, kernel_size, 1, num_outputs, 'conv1', activation_fn)
		h_2 = conv2d(h_1, kernel_size, 1, num_outputs, 'conv2', None)
		return input_ + h_2
