# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:14:38 2019

@author: DrLC
"""

import numpy as np

def fgsm(model, feed_dict, sess, epsilon=0.01, interval=[0, 1], norm=True):
    
    grad = sess.run(model.gradient(), feed_dict=feed_dict)[0]
    delta = epsilon * np.sign(grad)
    return delta

def apply_fgsm(x, delta, clip=True, interval=[0, 1]):
    
    x = np.clip(x + delta, interval[0], interval[1])
    return x

def flip(model, feed_dict, features, sess, dim=1433):
    
    grad = sess.run(model.gradient(), feed_dict=feed_dict)[0]
    delta_idx = np.argmax(-features*grad, 1)
    delta = np.eye(dim)[delta_idx.reshape(-1)]
    return delta

def apply_flip(x, delta):
    
    x = np.abs(np.sign(x) - delta)
    x_sum =  x.sum(1)[:, np.newaxis]
    x = x / (x_sum+1e-10)
    return x