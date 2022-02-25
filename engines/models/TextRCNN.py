from abc import ABC
import tensorflow as tf


class TextRCNN(tf.keras.Model, ABC):
    def __init__(self, configs, num_classes, vocab_size, embedding_matrix=None):
        super(TextRCNN, self).__init__()
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_dim = configs.embedding_dim
        self.hidden_dim = configs.hidden_dim
        self.dropout_rate = configs.dropout_rate
        if configs.embedding_method == "random":
            self.embedding = tf.keras.layers.Embedding(self.vocab_size + 1, self.embedding_dim, mask_zero=True)
        else:
            self.embedding = tf.keras.layers.Embedding(self.vocab_size + 1, self.embedding_dim, weights=[embedding_matrix], trainable=False)

        self.forward = tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True)
        self.backward = tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True, go_backwards=True)
        self.max_pool = tf.keras.layers.GlobalMaxPool1D()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate, name="dropout")
        self.dense1 = tf.keras.layers.Dense(2 * self.hidden_dim + self.embedding_dim, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                            bias_regularizer=tf.keras.regularizers.l2(0.1), name='dense')

    @tf.function
    def call(self, inputs, training=None):
        inputs = self.embedding(inputs)
        left_embedding = self.forward(inputs)
        right_embedding = self.backward(inputs)
        concat_outputs = tf.keras.layers.concatenate([left_embedding, inputs, right_embedding], axis=-1)
        dropout_outputs = self.dropout(concat_outputs, training)
        fc_outputs = self.dense1(dropout_outputs)
        pool_outputs = self.max_pool(fc_outputs)
        outputs = self.dense2(pool_outputs)
        return outputs
