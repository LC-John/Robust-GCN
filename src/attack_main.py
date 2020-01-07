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
from tqdm import tqdm

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

# Define attacking epoch
def attack(attacker, attacker_name, model, features, support, y, mask, placeholders, sess, **kw):

    mask_split = []
    results = []
    cost, acc = 0, 0
    for i in range(len(mask)):
        if mask[i] == True:
            mask_tmp = mask.copy()
            mask_tmp[:i] = False
            mask_tmp[i+1:] = False
            mask_tmp[i] = True
            mask_split.append(mask_tmp)
            results.append([])
    if attacker_name.lower() == 'fgsm':
        assert ('epsilon' in kw)
        for i in tqdm(range(len(mask_split))):
            results[i].append(fgsm(model=model,
                                   feed_dict=construct_feed_dict(features, support, y, mask_split[i], placeholders),
                                   features=features,
                                   sess=sess,
                                   epsilon=kw['epsilon']))
            cost_tmp, acc_tmp, _ = evaluate(results[i][-1], support, y, mask_split[i], placeholders)
            cost += cost_tmp
            acc += acc_tmp
    elif attacker_name.lower() == 'flip_single':
        assert ('n_flip' in kw)
        for i in tqdm(range(len(mask_split))):
            features_tmp = features.copy()
            for it in range(kw['n_flip']):
                features_tmp = flip_single(model=model,
                                           feed_dict=construct_feed_dict(features_tmp, support, y, mask_split[i], placeholders),
                                           features=features_tmp,
                                           sess=sess)
            results[i].append(features_tmp)
            cost_tmp, acc_tmp, _ = evaluate(results[i][-1], support, y, mask_split[i], placeholders)
            cost += cost_tmp
            acc += acc_tmp
    else:
        assert False
    return results, cost / len(mask_split), acc / len(mask_split)

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

print("Optimization Finished!")

save_path = "../model/cora_gcn"
model.load(save_path, sess)

print ("\nTest set size = {:d}".format(test_mask.sum()))

# Testing
test_cost, test_acc, test_duration = evaluate(features_dense, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
print()

# Single Flip
n_flips = 10
for n_flip in range(1, n_flips+1):
    features_dense_adv_list, cost_adv, acc_adv = attack(attacker=flip_single,
                                                        attacker_name="flip_single",
                                                        model=model,
                                                        features=features_dense,
                                                        support=support,
                                                        y=y_test,
                                                        mask=test_mask,
                                                        placeholders=placeholders,
                                                        sess=sess,
                                                        n_flip=n_flip)
    dist = 0
    for features_adv in features_dense_adv_list:
        dist += np.abs(features_dense-features_adv[-1]).sum()
    print("Flip({:d})".format(n_flip), "results:",
          "cost=", "{:.4f}".format(cost_adv),
          "accuracy=", "{:.4f}".format(acc_adv),
          "dist=", "{:.4f}".format(dist/len(features_dense_adv_list)))
    del features_dense_adv_list
print()

# FGSM
epsilons = [0.00333, 0.001, 0.000333, 0.0001, 0.0000333, 0.00001]
for epsilon in epsilons[::-1]:
    features_dense_adv_list, cost_adv, acc_adv = attack(attacker=fgsm,
                                                        attacker_name="fgsm",
                                                        model=model,
                                                        features=features_dense,
                                                        support=support,
                                                        y=y_test,
                                                        mask=test_mask,
                                                        placeholders=placeholders,
                                                        sess=sess,
                                                        epsilon=epsilon)
    dist = 0
    for features_adv in features_dense_adv_list:
        dist += np.abs(features_dense-features_adv[-1]).sum()
    print("FGSM({:.5f})".format(epsilon), "results:",
          "cost=", "{:.4f}".format(cost_adv),
          "accuracy=", "{:.4f}".format(acc_adv),
          "dist=", "{:.4f}".format(dist/len(features_dense_adv_list)))
    del features_dense_adv_list
print()