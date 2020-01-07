#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 18:59 2019/12/11
# @Author: Sijie Shen
# @File: adversarial_train.py
# @Project: RobustGCN

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import time, os, sys
import tensorflow as tf
import numpy as np
import argparse

from attack import *
from utils import load_data, construct_feed_dict
from utils import preprocess_features, preprocess_adj, chebyshev_polynomials
from models import GCN, MLP

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', action='store', type=str, default=None,
                    help='Adversarial attack method, should be fgsm or flip')
args = parser.parse_args()

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_dir', '../data', "Dataset directory string.")
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_string('gpu', '1', 'GPU selection.')
flags.DEFINE_string('method', args.method, 'Adversarial attack method')

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset_dir,
                                                                                   FLAGS.dataset)

# Some preprocessing
features_dense, features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

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
    feed_dict_val = construct_feed_dict(features_dense, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()

    if epoch >= 20:
        # Generate adversarial examples for adversarial training
        if args.method == 'flip':
            # Flip
            n_flip = 5
            features_dense_adv = features_dense
            for i in range(n_flip):
                features_dense_adv = flip(model=model,
                                          feed_dict=construct_feed_dict(features_dense_adv, support, y_test, test_mask,
                                                                        placeholders),
                                          features=features_dense_adv,
                                          sess=sess)

        elif args.method == 'fgsm':
            # FGSM
            epsilon = 0.1
            features_dense_adv = fgsm(model=model,
                                      feed_dict=construct_feed_dict(features_dense, support, y_test, test_mask,
                                                                    placeholders),
                                      features=features_dense,
                                      sess=sess,
                                      epsilon=epsilon)
        else:
            raise NotImplementedError
        # Random sample several training nodes and replace with adversarial examples
        adv_ratio = epoch / FLAGS.epochs
        train_range = 1708
        samples = np.random.choice(train_range, int(train_range * adv_ratio), replace=False)
        features_dense_mix = np.array(features_dense)
        features_dense_mix[samples] = features_dense_adv[samples]
    else:
        features_dense_mix = features_dense

    # Construct feed dictionary
    feed_dict = construct_feed_dict(features_dense_mix, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features_dense_mix, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

save_path = "../model/" + FLAGS.dataset + "_" + FLAGS.model + "/" + FLAGS.method
model.save(save_path, sess)

# Testing
test_cost, test_acc, test_duration = evaluate(features_dense, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
