# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:57:31 2017

@author: Florent
"""
import tensorflow as  tf
import numpy as np

def unif_weight_init (shape,name=None):
     '''Initializes the Weight Variables according to the Glorot-Bengio  Rule'''
     initial = tf.random_uniform(shape, minval=-np.sqrt(6.0/(shape[0]+shape[1])), maxval=np.sqrt(6.0/(shape[0]+shape[1])), dtype=tf.float32)
     return tf.Variable(initial, name=name)
 
def sample_gaussian (mean,diag_cov):
    '''Samples a multivariate gaussian with the given  mean and diagonal covariance'''
    z = mean + tf.random_normal(diag_cov.shape) * diag_cov
    return z
def sample_gaussian_np (mean,diag_cov):
    '''Samples a multivariate gaussian with the given  mean and diagonal covariance'''
    z = mean + np.random.normal(size=diag_cov.shape) * diag_cov
    return z

def sigmoid (x):
    return 1.0/(1.0+np.exp(-x))