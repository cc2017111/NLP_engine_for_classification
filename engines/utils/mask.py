import tensorflow as tf


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # [batch_size, 1, 1, seq_len]
    # attention: attention_matrix=[batch_size, num_head, seq_len_q, seq_len_k]
    return seq[:, tf.newaxis, tf.newaxis, :]
