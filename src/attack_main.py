# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:41:14 2019

@author: DrLC
"""

from __future__ import division
from __future__ import print_function

import time, os, sys
import tensorflow as tf
import numpy as np

from attack import fgsm, flip, flip_single

from utils import load_data, construct_feed_dict
from utils import preprocess_features, preprocess_adj
from models import GCN

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_string('gpu', '1', 'GPU selection.')

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('../data', 'cora')

# Some preprocessing
features_dense, features = preprocess_features(features)
support = [preprocess_adj(adj)]
num_supports = 1
model_func = GCN

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32, shape=features[2]),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)
# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)
 

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

print("Optimization Finished!")

save_path = "../model/cora_gcn"
model.load(save_path, sess)

print ("Test set size = {:d}\n".format(test_mask.sum()))

# Testing
test_cost, test_acc, test_duration = evaluate(features_dense, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
print()

# FGSM
epsilons = [0.1, 0.03, 0.01, 0.003, 0.001]
for epsilon in epsilons[::-1]:
    features_dense_adv = fgsm(model=model,
                              feed_dict=construct_feed_dict(features_dense, support, y_test, test_mask, placeholders),
                              features=features_dense,
                              sess=sess,
                              epsilon=epsilon)
    test_cost, test_acc, test_duration = evaluate(features_dense_adv, support, y_test, test_mask, placeholders)
    print("FGSM({:.3f})".format(epsilon), "results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc),
          "dist=", "{:.1f}".format(np.abs(features_dense-features_dense_adv).sum()),
          "time=", "{:.5f}".format(test_duration))
print()
    
# Flip
n_flip = 5
features_dense_adv = features_dense
for i in range(n_flip):
    features_dense_adv = flip(model=model,
                              feed_dict=construct_feed_dict(features_dense_adv, support, y_test, test_mask, placeholders),
                              features=features_dense_adv,
                              sess=sess)
    test_cost, test_acc, test_duration = evaluate(features_dense_adv, support, y_test, test_mask, placeholders)
    print("Flip({:d})".format(i+1), "results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc),
          "dist=", "{:.1f}".format(np.abs(features_dense-features_dense_adv).sum()),
          "time=", "{:.5f}".format(test_duration))
print()

# Single Flip
n_flip = 1000
features_dense_adv = features_dense
for i in range(n_flip):
    features_dense_adv = flip_single(model=model,
                                     feed_dict=construct_feed_dict(features_dense_adv, support, y_test, test_mask, placeholders),
                                     features=features_dense_adv,
                                     sess=sess)
    test_cost, test_acc, test_duration = evaluate(features_dense_adv, support, y_test, test_mask, placeholders)
    print("SingleFlip({:d})".format(i+1), "results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc),
          "dist=", "{:.1f}".format(np.abs(features_dense-features_dense_adv).sum()),
          "time=", "{:.5f}".format(test_duration))
print()