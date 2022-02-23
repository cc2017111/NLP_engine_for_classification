from abc import ABC
import tensorflow as tf


class TextRNN(tf.keras.Model, ABC):
    def __init__(self, configs, num_classes, vocab_size, embedding_metrix=None):
        super(TextRNN, self).__init__()
        self.embedding_dim = configs.embedding_dim
        self.num_classes = num_classes
        self.hidden_dim = configs.hidden_dim
        self.vocab_size = vocab_size
        self.dropout_rate = configs.dropout_rate
        self.use_attention = configs.use_attention
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True))
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate, name='dropout')
        if self.use_attention:
            self.attention_w = tf.Variable(tf.zeros([1, 2 * self.hidden_dim]))
        self.dense = tf.keras.layers.Dense(self.num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                           bias_regularizer=tf.keras.regularizers.l2(0.1), name='dense')
        if configs.embedding_method == "random":
            print("ok")
            self.embedding_layer = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim,
                                                             mask_zero=True)
        else:
            self.embedding_layer = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim,
                                                             weights=[self.embedding_matrix], trainable=False)

    @tf.function
    def call(self, inputs, training=None):
        inputs = self.embedding_layer(inputs)
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
