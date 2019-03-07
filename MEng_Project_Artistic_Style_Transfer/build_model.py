from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import normalization as nl 
import PIL.Image
import numpy as np
'''
it contains:
      loss function
      model_transform function
      style prediction function
      inception_v3 network
'''
import loss_function as losses
import model_transform as transformer_model
from StylePrediction import style_prediction, _inception_v3_arg_scope
from tensorflow.contrib.slim.python.slim.nets import inception_v3
import image_utils

slim = tf.contrib.slim



def build_model(  content_input_,
                  style_input_,
                  trainable,
                  is_training,
                  reuse = None,
                  inception_end_point = 'Mixed_6e',
                  style_prediction_bottleneck = 100,
                  adds_losses = True,
                  content_weights = None,
                  style_weights = None,
                  total_variation_weight = None
          ):
  

    '''
It returns the scope name
'residual/residual1/conv1',
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
'expand/conv3/conv'
And the depths [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 64, 32, 3]
    '''

    [activation_names, activation_depths] = transformer_model.style_normalization_activations() 
  
  # Style_prediciton will be defined later
    style_params, bottleneck_feat = style_prediction( 
                      style_input_,
                      activation_names,
                      activation_depths,
                      is_training = is_training,
                      inception_end_point = inception_end_point,
                      style_prediction_bottleneck = style_prediction_bottleneck,
                      reuse = reuse
                            )

    stylized_images = transformer_model.transform(
                              content_input_,
                              normalizer_fn=nl.conditional_style_norm, # This is the
                              reuse=reuse,                      # part we can
                              trainable=trainable,              # use Group 
                              is_training=is_training,          # Normalization
                              normalizer_params= {'style_params': style_params}
                              )


    loss_dict = {}
    total_loss = []

    # I think this if statement totally waster time
    if adds_losses:
        total_loss, loss_dict = losses.total_loss(
                                        content_input_,
                                        style_input_,
                                        stylized_images,
                                        content_weights=content_weights,
                                        style_weights=style_weights,
                                        total_variation_weight=total_variation_weight
                                        )

    return stylized_images, total_loss, loss_dict, bottleneck_feat





'''
# Because it has to have train dataset
# it cannot be just called from the build model
a = build_model(
                content_input_=x,
                style_input_ =y,
                trainable=False,
                is_training=False,
                reuse=None,
                inception_end_point='Mixed_6e',
                style_prediction_bottleneck=100,
                adds_losses=None,
                content_weights=None,
                style_weights=None,
                total_variation_weight=1e4
                )
'''