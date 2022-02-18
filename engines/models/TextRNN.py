from abc import ABC
import tensorflow as tf


class TextRNN(tf.keras.Model, ABC):
    def __init__(self, configs, num_classes, hidden_dim, embedding_dim, vocab_size, embedding_metrix=None):
        super(TextRNN, self).__init__()
        if configs.embedding_method is "random":
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_metrix], trainable=False)

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.dropout_rate = configs.dropout_rate
        self.use_attention = configs.use_attention
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True))
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate, name='dropout')
        if self.use_attention:
            self.attention_w = tf.Variable(tf.zeros([1, 2 * self.hidden_dim]))
        self.dense = tf.keras.layers.Dense(self.hidden_dim, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                           bias_regularizer=tf.keras.regularizers.l2(0.1), name='dense')

    @tf.function
    def call(self, inputs, training=None):
        inputs = self.embedding(inputs)
        bilstm_outputs = self.bilstm(inputs)
        if self.use_attention:
            outputs = tf.nn.tanh(bilstm_outputs)
            outputs = tf.matmul(outputs, self.attention_w, transpose_b=True)
            alpha = tf.nn.softmax(outputs, axis=1)
            outputs = alpha * bilstm_outputs
            bilstm_outputs = tf.nn.tanh(outputs)
        dropout_outputs = self.dropout(bilstm_outputs, training)
        outputs = tf.reduce_sum(dropout_outputs, axis=1)
        outputs = self.dense(outputs)
        return outputs
