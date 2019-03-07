from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import loss_gram as learning_utils
import vgg


# The most important question is that how do we know the
# data type and shape of the content_weights at first 

def content_loss(	end_points, 
					stylized_end_points,
					content_weights):

	tota_content_loss = np.float32(0.0)
	content_loss_dict = {}

	# content_weights will contain :{ 'name': values }

	for name, weight in  content_weights.iteritems():
		
		# Not sure what values does the end_points contain 
		loss = tf.reduce_mean(( end_points[name] - stylized_end_points[name] )**2)

		weighted_loss = weight * loss

		content_loss_dict['content_loss/' + name] = loss
		content_loss_dict['weighted_content_loss/' + name] = weighted_loss

		total_content_loss += weighted_loss

	content_loss_dict ['total_content_loss'] = total_content_loss

	return total_content_loss, content_loss_dict


def style_loss(style_end_points,
			   stylized_end_points,
			   style_weights):

	total_style_loss = np.float32(0.0)
	style_loss_dict = {}

	for name, weight in style_weights.iteritems():
		loss = tf.reduce_mean(
			(learning_utils.gram_matrix(stylized_end_points[name]) -
			learning_utils.gram_matrix(style_end_points[name])) ** 2
			)

		weighted_loss = weight * loss

		style_loss_dict['style_loss/' + name] = loss
		style_loss_dict['weighted_style_loss/' + name] = weighted_loss
		total_style_loss += weighted_loss

	style_loss_dict['total_style_loss'] = total_style_loss

	return total_style_loss, style_loss_dict



def total_loss(	content_inputs,
				 style_inputs,
				 stylized_input,
				 content_weights,
				 style_weights,
				 total_variation_weight,
				 reuse = False):

	with tf.name_scope('content_endpoints'):
		content_end_points = vgg.vgg_16(content_inputs, reuse = reuse)

	with tf.name_scope('style_endpoints'):
		style_end_points = vgg.vgg_16(style_inputs, reuse = reuse)

	with tf.name_scope('stylized_endpoints'):
		stylized_end_points = vgg.vgg_16(stylized_input, reuse = reuse)


	# compute the content loss

	with tf.name_scope('content_loss'):
		total_cotent_loss, content_loss_dict = content_loss(	content_end_points,
									stylized_end_points,
									content_weights
																)
	# compute the style loss

	with tf.name_scope('style_loss'):
		total_style_loss, style_loss_dict = content_loss(	style_end_points,
									stylized_end_points,
									style_weights
																)
	# Compute the total variation loss
	with tf.name_scope('total_variation_loss'):
		tv_loss, total_variation_loss_dict = learning_utils.total_variation_loss(
									stylized_inputs,
									total_variation_weight)
	# compute the total loss

	with tf.name_scope('total_loss'):
		loss = total_cotent_loss + total_style_loss + tv_loss
	

	# Explain how does the update work
	'''
	d = {1: "one", 2: "three"}
	d1 = {2: "two"}

	# updates the value of key 2
	d.update(d1)
	print(d)

	d1 = {3: "three"}

	# adds element with key 3
	d.update(d1)
	print(d)


	{1: 'one', 2: 'two'}
	{1: 'one', 2: 'two', 3: 'three'}
	
	'''

	loss_dict = {'total_loss': loss}
	loss_dict.update(content_loss_dict)
	loss_dict.update(style_loss_dict)
	loss_dict.update(total_variation_loss_dict)

	return loss, loss_dict

