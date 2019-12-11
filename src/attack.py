# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:14:38 2019

@author: DrLC
"""

import numpy as np

def fgsm(model, feed_dict, features, sess, epsilon=0.01, interval=[0, 1], norm=True):
    
    grad = sess.run(model.gradient(), feed_dict=feed_dict)[0]
    delta = epsilon * np.sign(grad)
    x = np.clip(features + delta, interval[0], interval[1])
    if norm:
        x = x / (x.sum(1)[:, np.newaxis]+1e-10)
    return x

def flip(model, feed_dict, features, sess, dim=1433):
    
    grad = sess.run(model.gradient(), feed_dict=feed_dict)[0]
    delta_idx = np.argmax(-features*grad, 1)
    delta = np.eye(dim)[delta_idx.reshape(-1)]
    x = np.abs(np.sign(features) - delta)
    x_sum =  x.sum(1)[:, np.newaxis]
    x = x / (x_sum+1e-10)
    return x

def flip_single(model, feed_dict, features, sess, dim=1433):
    
    grad = sess.run(model.gradient(), feed_dict=feed_dict)[0]
    delta_idx_col = np.argmax(-features*grad, 1)
    delta_val_candidate = np.max(-features*grad, 1)
    delta_idx_row = np.argmax(delta_val_candidate)
    delta = np.zeros(grad.shape)
    delta[delta_idx_row][delta_idx_col[delta_idx_row]] = 1
    x = np.abs(np.sign(features) - delta)
    x_sum =  x.sum(1)[:, np.newaxis]
    x = x / (x_sum+1e-10)
    return x