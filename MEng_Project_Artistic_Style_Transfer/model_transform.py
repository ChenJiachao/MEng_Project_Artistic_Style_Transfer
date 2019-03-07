from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import  model_transform_components as model_util

slim = tf.contrib.slim


def transform( input_,
		normalizer_fn = None,
		normalizer_params = None,
		reuse = False,
		trainable = True,
		is_training = True
				):

	with tf.variable_scope('transformer', reuse = reuse):
		# There are multy layers of arg_scope but I CERTAINLY
		# DON'T KNOW WHY 


		# this is for the convolution 2 D with batch normalization

		with slim.arg_scope(	
					[slim.conv2d],
					activation_fn = tf.nn.relu,
					normalizer_fn = normalizer_fn,
					normalizer_params = normalizer_params,
					weights_initializer=tf.random_normal_initializer(0.0, 0.01),
					biases_initializer=tf.constant_initializer(0.0),
					trainable=trainable
								):
			with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                          normalizer_params=None,
                          trainable=trainable):
				with slim.arg_scope( 	
							[slim.batch_norm],
							is_training = is_training,
							trainable = trainable
									):

				# I still think it is for convolution 
					with tf.variable_scope( 'contract'):
						h = model_util.conv2d(input_, 9, 1, 32, 'conv1')
						h = model_util.conv2d(h, 3, 2, 64, 'conv2')
						h = model_util.conv2d(h, 3, 2, 128, 'conv3')
			with tf.variable_scope('residual'):
				h = model_util.residual_block(h, 3, 'residual1')
				h = model_util.residual_block(h, 3, 'residual2')
				h = model_util.residual_block(h, 3, 'residual3')
				h = model_util.residual_block(h, 3, 'residual4')
				h = model_util.residual_block(h, 3, 'residual5')
			with tf.variable_scope('expand'):
				h = model_util.upsampling(h, 3, 2, 64, 'conv1')
				h = model_util.upsampling(h, 3, 2, 32, 'conv2')
				return model_util.upsampling(h, 9, 1, 3, 'conv3', 
										activation_fn=tf.nn.sigmoid)



def style_normalization_activations(pre_name='transformer',
                                    post_name='StyleNorm'):

	scope_names = ['residual/residual1/conv1',
				 'residual/residual1/conv2',
				 'residual/residual2/conv1',
				 'residual/residual2/conv2',
				 'residual/residual3/conv1',
				 'residual/residual3/conv2',
				 'residual/residual4/conv1',
				 'residual/residual4/conv2',
				 'residual/residual5/conv1',
				 'residual/residual5/conv2',
				 'expand/conv1/conv',
				 'expand/conv2/conv',
				 'expand/conv3/conv']

	scope_names = ['{}/{}/{}'.format(pre_name, name, post_name)
					for name in scope_names]

	depths = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 64, 32, 3]

	return scope_names, depths
