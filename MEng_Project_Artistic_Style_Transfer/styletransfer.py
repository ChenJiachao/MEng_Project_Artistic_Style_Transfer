from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import os

import numpy as np
import tensorflow as tf

import build_model as build_model
import image_utils
from PIL import Image
slim = tf.contrib.slim


style_images_paths = 'style7.jpeg'
content_images_paths = 'content/me3/me3255/*.jpg'
checkpoints_path = 'model.ckpt'
output_dir = 'output/me3/'
image_size = 255	
style_image_size = 255
maximum_styles_to_evaluate = 1024
content_square_crop = False
style_square_crop = False
interpolation = '[0.8]'



def style_transfer(style_images_paths,
					content_images_paths,
					checkpoints_path,
					output_dir,
					image_size,
					style_image_size,
					maximum_styles_to_evaluate,
					content_square_crop,
					style_square_crop,
					interpolation):

	tf.logging.set_verbosity(tf.logging.INFO)
	if not tf.gfile.Exists(output_dir):
		tf.gfile.MkDir(output_dir)

	with tf.Graph().as_default(), tf.Session() as sess:
		# Defines place holder for the style image.

		style_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
		
		if style_square_crop:
			style_img_preprocessed = image_utils.center_crop_resize_image( style_img_ph,style_image_size)
		else:
			style_img_preprocessed = image_utils.resize_image(style_img_ph, style_image_size)
		
		

		# Defines place holder for the content image.
		content_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
		
		if content_square_crop:
			content_img_preprocessed = image_utils.center_crop_resize_image(
					content_img_ph, image_size)
		else:
			content_img_preprocessed = image_utils.resize_image(
					content_img_ph, image_size)
		
		
		# Defines the model.
		stylized_images, _, _, bottleneck_feat = build_model.build_model(
				content_img_preprocessed,
				style_img_preprocessed,
				trainable=False,
				is_training=False,
				inception_end_point='Mixed_6e',
				style_prediction_bottleneck=100,
				adds_losses=False)

		if tf.gfile.IsDirectory(checkpoints_path):
			checkpoint = tf.train.latest_checkpoint(checkpoints_path)
		else:
			checkpoint = checkpoints_path
			tf.logging.info('loading latest checkpoint file: {}'.format(checkpoint))

		init_fn = slim.assign_from_checkpoint_fn(checkpoint,
																						 slim.get_variables_to_restore())
		sess.run([tf.local_variables_initializer()])
		init_fn(sess)

		# Gets the list of the input style images.
		style_img_list = tf.gfile.Glob(style_images_paths)
		if len(style_img_list) > maximum_styles_to_evaluate:
			np.random.seed(1234)
			style_img_list = np.random.permutation(style_img_list)
			style_img_list = style_img_list[:maximum_styles_to_evaluate]

		# Gets list of input content images.
		content_img_list = tf.gfile.Glob(content_images_paths)

		for content_i, content_img_path in enumerate(content_img_list):
			content_img_np = image_utils.load_np_image_uint8(content_img_path)[:, :, : 3]
			content_img_name = os.path.basename(content_img_path)[:-4]

			# Saves preprocessed content image.
			inp_img_croped_resized_np = sess.run(
					content_img_preprocessed, feed_dict={
							content_img_ph: content_img_np
					})
			#image_utils.save_np_image(inp_img_croped_resized_np, os.path.join(output_dir, s.jpg' % (content_img_name)))

			# Computes bottleneck features of the style prediction network for the
			# identity transform.
			identity_params = sess.run(
					bottleneck_feat, feed_dict={style_img_ph: content_img_np})

			for style_i, style_img_path in enumerate(style_img_list):
				if style_i > maximum_styles_to_evaluate:
					break
				style_img_name = os.path.basename(style_img_path)[:-4]
				style_image_np = image_utils.load_np_image_uint8(style_img_path)[:, :, : 3]

				if style_i % 10 == 0:
					tf.logging.info('Stylizing (%d) %s with (%d) %s' %
													(content_i, content_img_name, style_i,
													 style_img_name))

				# Saves preprocessed style image.
				style_img_croped_resized_np = sess.run(
						style_img_preprocessed, feed_dict={
								style_img_ph: style_image_np
						})
				#image_utils.save_np_image(style_img_croped_resized_np, os.path.join(output_dir,'%s.jpg' % (style_img_name)))

				# Computes bottleneck features of the style prediction network for the
				# given style image.
				style_params = sess.run(
						bottleneck_feat, feed_dict={style_img_ph: style_image_np})

				interpolation_weights = ast.literal_eval(interpolation)
				# Interpolates between the parameters of the identity transform and
				# style parameters of the given style image.
				for interp_i, wi in enumerate(interpolation_weights):
					stylized_image_res = sess.run(
							stylized_images,
							feed_dict={
									bottleneck_feat:
											identity_params * (1 - wi) + style_params * wi,
									content_img_ph:
											content_img_np
							})

					# Saves stylized image.
					
				image_utils.save_np_image(
							stylized_image_res,
							os.path.join(output_dir, '%s_stylized_%d.jpg' %
													 (content_img_name,interp_i)))
					


				#stylized_image_res = np.uint8(stylized_image_res * 255.0)
				#img = np.squeeze(stylized_image_res, 0)
				#print(img)
				#img = Image.fromarray(img, 'RGB')
				


style_transfer(		style_images_paths,
					content_images_paths,
					checkpoints_path,
					output_dir,
					image_size,
					style_image_size,
					maximum_styles_to_evaluate,
					content_square_crop,
					style_square_crop,
					interpolation)



'''	
def console_entry_point():
	tf.app.run(main)

if __name__ == '__main__':
	console_entry_point()
'''