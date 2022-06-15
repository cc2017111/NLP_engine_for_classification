import math

import numpy as np
import pandas as pd
import tensorflow as tf


# L(pt) = -at(1 - pt)^y log(pt)
# if y = 1 pt = p, at = a
# if y = 0 pt = 1 - p at = 1 - a
class FocalLoss:
    def __init__(self, is_alpha=True, alpha=0.25, gamma=2, epsilon=1e-9):
        self.is_alpha = is_alpha
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def call(self, y_probs, y_true):
        positive_pt = tf.where(tf.equal(y_true, 1), y_probs, tf.ones_like(y_probs))
        negative_pt = tf.where(tf.equal(y_true, 0), 1 - y_probs, tf.ones_like(y_probs))
        if self.is_alpha:
            loss = -self.alpha * tf.pow(1 - positive_pt, self.gamma) * tf.math.log(tf.clip_by_value(positive_pt, self.epsilon, 1.)) - \
                    (1 - self.alpha) * tf.pow(1 - negative_pt, self.gamma) * tf.math.log(tf.clip_by_value(negative_pt, self.epsilon, 1.))
            return tf.reduce_sum(loss, axis=1)
        else:
            loss = -tf.pow(1 - positive_pt, self.gamma) * tf.math.log(tf.clip_by_value(positive_pt, self.epsilon, 1.)) - \
                    tf.pow(1 - negative_pt, self.gamma) * tf.math.log(tf.clip_by_value(negative_pt, self.epsilon, 1.))
            return tf.reduce_mean(loss, axis=1)


# y_true = tf.constant([[0, 1, 0], [0, 1, 0]])
# y_probs = tf.constant([[0.25, 0.5, 0.25], [0.1, 0.8, 0.1]])
# focal_loss = FocalLoss(is_alpha=True, alpha=2.5, gamma=1.5, epsilon=1e-6)
# print(focal_loss.call(y_true, y_probs))
# loss = tf.keras.losses.CategoricalCrossentropy()
# print(loss.call(y_true, y_probs))
# print(loss.call(y_true, y_probs))