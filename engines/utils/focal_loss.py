import numpy as np
import pandas as pd
import tensorflow as tf


# L(pt) = -at(1 - pt)^y log(pt)
# if y = 1 pt = p, at = a
# if y = -1 pt = 1 - p at = 1 - a
def focal_loss(y_true, y_probs, alpha=0.5, gamma=1.5, epsilon=1e-6):
    positive_pt = tf.where(tf.equal(y_true, 1), y_probs, tf.ones_like(y_probs))
    negative_pt = tf.where(tf.equal(y_true, 0), 1 - y_probs, tf.ones_like(y_probs))
    loss = -alpha * tf.pow(1 - positive_pt, gamma) * tf.math.log(tf.clip_by_value(positive_pt, epsilon, 1.)) - \
           (1 - alpha) * tf.pow(1 - negative_pt, gamma) * tf.math.log(tf.clip_by_value(negative_pt, epsilon, 1.))
    return tf.reduce_mean(loss, axis=1)
