from abc import ABC
import tensorflow as tf


class FastText(tf.keras.Model, ABC):
    def __init__(self, configs, num_classes, vocab_size, embedding_matrix=None):
        super(FastText, self).__init__()
        self.num_clasess = num_classes
        self.vocab_size = vocab_size
        self.embedding_dim = configs.embedding_dim
        if configs.embedding_method == 'random':
            self.embedding = tf.keras.layers.Embedding(self.vocab_size + 1, self.embedding_dim, mask_zero=True)
        else:
            self.embedding = tf.keras.layers.Embedding(self.vocab_size + 1, self.embedding_dim, weights=[embedding_matrix], trainable=False)
        self.avg_pool = tf.keras.layers.GlobalAvgPool1D()
        self.dense = tf.keras.layers.Dense(self.num_clasess, activation='softmax')

    @tf.function
    def call(self, inputs, training=None):
        embedding_outputs = self.embedding(inputs)
        pool_outputs = self.avg_pool(embedding_outputs)
        outputs = self.dense(pool_outputs)
        return outputs
