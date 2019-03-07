# Origianally, is the ops.py file
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.ops import variable_scope

slim = tf.contrib.slim


@slim.add_arg_scope


def conditional_instance_norm( inputs,
								labels,
								num_categories, # Total number of styles be modeled
								center = True,	# If true, substract beta
								scale = True,	# If true, add gamma
								activation_fn = None,
								reuse = None,
								variables_collections = None, 
								outputs_collections = None,	# Collection to add outputs
								trainable = True,
								scope = None):	# Collections: because disconnected parts of 
												# a TensorFlow program might want to create 
												# variables, it is sometimes useful to have 
												# a single way to access all of them. 
												# For this reason TensorFlow provides 
												# collections, which are named lists of 
												# tensors or other objects, 
												#such as tf.Variable instances.
	with tf.variable_scope( scope,
							'InstanceNorm',
							[inputs],
							reuse = reuse) as sc:
		# The tensorflow part will begin with the " with tf.variable_scope()"
		# First, we need to convert our inputs to tensors
		# Second, we get its shape
		# Third, we get its dimension or rank
		inputs = tf.convert_to_tensor(inputs)	# Convert given values to tensors
		inputs_shape = inputs.get_shape()
		inputs_rank = inputs_shape.ndims


		# The reason why we need to get the rank of the inputs
		# is that we need to make sure our tensor is a 4 D tensor
		# as be concerned with the normalization facts
		if inputs_rank is None:
			raise ValueError('Inputs %s has undefined rank' % inputs.name)
		if inputs_rank != 4:
			raise ValueError(' Inputs %s is not a 4 D tensor' % inputs.name)


		# dtype.base_dtype return a non-reference Dtype based on this Dtype
		dtype = inputs.dtype.base_dtype
		axis = [1, 2]
		params_shape = inputs_shape[-1:]	# get the last value of inputs_shape

		if not params_shape.is_fully_defined():
			raise ValueError('Inputs %s has undefined last dimension %s. % '% ( inputs.name, 
																				params_shape) )

		#This function will get gamma and beta
		def _label_conditioned_variable(
										name,
										initializer,
										labels,
										num_categories
										):

			# Don't know why do we need to concatenate 
			shape = tf.TensorShape([num_categories].concatenate(params_shape))

			
			# I guess it is away to get variable collection in slim
			# for shrinking the code lines.
			var_collections = slim.utils.get_variable_collections(	variables_collections, 
																	name)

			# The way how does slim work, to shrink the size of the code
			var = slim.model_variable(	name,
										shape = shape,
										dtype = dtype,
										initializer = initializer,
										collections = var_collections,
										trainable = trainable)
			# tf.gather will select the variables from the given indices
			conditioned_var = tf.gather(var, labels)
			
			
			# expand_dims can help us increase the dimension by 1
			# for the images, it is 2-D, while we use normalization 
			# we need have 4-D, thats why we increase it twice, from the 
			# the front and end.

			conditioned_var = tf.expand_dims(tf.expand_dims(conditioned_var,1),1)

			return conditioned_var


		# Let's begin to the normalizaiton part
		# which contains beta, gamma as the equation demonstrates
		beta, gamma = None, None

		
		# Go back to the previous paramaters set up in the conditioned
		# normalization, center = True -> substract the beta
		if center: 
			beta = _label_conditioned_variable(	'beta',
												tf.zeros_initializer(),
												labels,
												num_categories)
		

		# Same explaination to center
		if scale:
			gamma = _label_conditioned_variable( 	'gamma',
													tf.ones_initializer(),
													labels,
													num_categories)
		

		# The inputs will be [batch, height, width, channels]
		# axis= [1, 2], wil be the height and width
		# tf.nn.moments() is the API to calculate the mean and vairance
		mean, variance = tf.nn.moments(inputs, axis, keep_dims = True)

		# Calculate the instance normalization while still need borrow
		# batch normalization


		#the variance epsilon - a small float number to avoid dividing by 0
		variance_epsilon = 1E-5

		outputs = tf.nn.batch_normalization( 	inputs,
												mean,
												variance,
												beta,
												gamma,
												variance_epsilon)
		# Again, no idea what it is
		outputs.set_shape(inputs_shape)

		if activation_fn:
			outputs = activation_fn(outputs)

		return slim.utils.collect_named_outputs(outputs_collections,
												sc.original_name_scope,
												outputs)



@slim.add_arg_scope

def conditional_style_norm(inputs,
							style_params = None,
							activation_fn =None,
							reuse = None,
							outputs_collections = None,
							check_numerics = True,
							scope = None
							):
	# All the way it should go
	with variable_scope.variable_scope( scope,
										'StyleNorm',
										[inputs],
										reuse = reuse
										) as sc:
		
		inputs = framework_ops.convert_to_tensor(inputs)
		inputs_shape = inputs.get_shape()
		inputs_rank = inputs_shape.ndims
		
		# just the way to check... so tired about these code
		if inputs_rank is None :
			raise ValueError('Inputs %s has undefined rank.'% inputs.name)
		if inputs_rank != 4 :
			raise ValueError('Inputs %s is not a 4D tensor.'% inputs.name) 
		
		# Heighs and Widths ....
		axis = [1, 2]
		params_shape = inputs_shape[-1:]

		if not params_shape.is_fully_defined():
			raise ValueError('Inputs %s has undefined last dimension %s.' % (inputs.name, params_shape))



		def _style_parameters(name):
			var = style_params[('{}/{}'.format(sc.name, name))]

			if check_numerics:
				var = tf.check_numerics(var, 'NaN/Inf in {}'.format(var.name))
			if var.get_shape().ndims < 2:
				var = tf.expand_dims(var, 0)
			var = tf.expand_dims(tf.expand_dims(var, 1), 1)

			return var

		# Allocates parameters for the beta and gamma of the normalization.
		beta = _style_parameters('beta')
		gamma = _style_parameters('gamma')

		# Calculates the moments on the last axis (instance activations).
		mean, variance = tf.nn.moments(	
									inputs, 
									axis, 
									keep_dims=True
									)

		# Compute layer normalization using the batch_normalization function.
		variance_epsilon = 1E-5
		outputs = tf.nn.batch_normalization(
										inputs, 
										mean, 
										variance,
										beta, 
										gamma,
										variance_epsilon
										)
		# set_shape like reshape, but it only update the shape information not change them
		# which means it cannot directly change the shape
		outputs.set_shape(inputs_shape)
	
		if activation_fn:
			outputs = activation_fn(outputs)
	
		# It will change the name of the variable
		# and return to the collection
		return slim.utils.collect_named_outputs(
											outputs_collections,
											sc.original_name_scope, 
											outputs
											)