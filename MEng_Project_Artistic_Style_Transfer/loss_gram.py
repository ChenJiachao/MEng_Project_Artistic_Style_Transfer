from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

import vgg

slim = tf.contrib.slim

def precompute_gram_matrices( image, 
                final_endpoint = 'fc8'
                ):
  with tf.Session() as session:
    end_points = vgg.vgg16( image, final_endpoint = final_endpoint)


    # In vgg file, we have tf.variable_scope('vgg_16', [inputs], reuse=reuse) as sc:
    # In vgg file, we have variable scope named 'vgg_16'
    tf.train.Saver( slim.get_variable( 'vgg_16')).restore( session, 
                                vgg.checkpoint_file)

    # The gram_matrix function will be defined later
    # The end_points is in the vgg.py file
    # it contains the layers' name (like conv1) and values.

    return dict(
          [(key, gram_matrix(value).eval()) 
          for key, value in end_points.iteritems()]
          )


def gram_matrix(feature_maps):

  # the feature map tensor shape should [A,B,C,D]
  # while use tf.unstack, it will be A,B,C,D
  '''
  import tensorflow as tf

  a = tf.constant ([1,2,3,4])
  b = tf.unstack(a, axis = 0)
  q,w,e,r = b
  with tf.Session() as sc:
    print(sc.run(b))
    print(sc.run(q))
    output:
    [1, 2, 3, 4]
  1
  '''
  batch_size, height, width, channels = tf.unstack(tf.shape(feature_maps))
  denominator = tf.to_float(height * width)

    # in this tf.stack, we will create the dimension of the cube of the batch
    
  feature_maps = tf.reshape(  feature_maps, 
                  tf.stack([batch_size, height * width, channels]))
  matrix = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
  return matrix / denominator



def total_variation_loss( stylized_inputs, 
              total_variation_weight):

    shape = tf.shape(stylized_inputs)
    
    # This shouldn't be talked too much
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]



    y_size = tf.to_float((height - 1) * width * channels)
    x_size = tf.to_float(height * (width - 1) * channels)
    
    # I dont know why
    # stylized_inputs[:, 1:, :, :] wouldn't contain the first element
    # stylized_inputs[:, :-1, :, :] wouldn't contain the last element
    
    y_loss = tf.nn.l2_loss(stylized_inputs[:, 1:, :, :] - stylized_inputs[:, :-1, :, :]) / y_size
    
    x_loss = tf.nn.l2_loss(stylized_inputs[:, :, 1:, :] - stylized_inputs[:, :, :-1, :]) / x_size
    
    loss = (y_loss + x_loss) / tf.to_float(batch_size)
    
    weighted_loss = loss * total_variation_weight
    
    return weighted_loss, {
      'total_variation_loss': loss,
      'weighted_total_variation_loss': weighted_loss
  }
