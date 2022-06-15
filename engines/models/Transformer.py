from abc import ABC
import tensorflow as tf
import numpy as np
from utils.mask import create_padding_mask


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, sequence_length):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.pe = np.array([[pos / np.power(10000, 2 * (i // 2) / self.embedding_dim)
                             for i in range(self.embedding_dim)] for pos in range(self.sequence_length)])

        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])

    @tf.function
    def call(self, inputs):
        position_embedding = inputs + self.pe
        return position_embedding


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, head_num):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num

        if self.embedding_dim % self.head_num != 0:
            raise ValueError('embedding_dim({}) % head_num({}) is not zero. embedding_dim must be multiple of head_num.'.format(
                self.embedding_dim, self.head_num))

        self.head_dim = self.embedding_dim // self.head_num

        self.W_Q = tf.keras.layers.Dense(self.embedding_dim)
        self.W_K = tf.keras.layers.Dense(self.embedding_dim)
        self.W_V = tf.keras.layers.Dense(self.embedding_dim)
        self.W_O = tf.keras.layers.Dense(self.embedding_dim)

    def scale_dot_product_attention(self, query, key, value, mask):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        print("matmul_qk:", matmul_qk)
        dk = self.head_num ** 0.5
        print(self.head_num)
        print(dk)
        scaled_attention = matmul_qk / tf.math.sqrt(dk)
        print(scaled_attention)

        if mask is not None:
            scaled_attention += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output

    def split_head(self, tensor, batch_size):
        tensor = tf.reshape(tensor, (batch_size, -1, self.head_num, self.head_dim))
        tensor = tf.transpose(tensor, perm=[0, 2, 1, 3])
        return tensor

    @tf.function
    def call(self, inputs, mask=None):
        print("inputs:", inputs)
        batch_size = tf.shape(inputs)[0]
        query = self.W_Q(inputs)
        print("query:", query)
        key = self.W_K(inputs)
        value = self.W_V(inputs)

        query = self.split_head(query, batch_size)
        key = self.split_head(key, batch_size)
        value = self.split_head(value, batch_size)

        z = self.scale_dot_product_attention(query, key, value, mask)
        z = tf.reshape(z, (batch_size, -1, self.embedding_dim))
        z = self.W_O(z)
        return z


class PositionWiseFeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, embedding_dim):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dense_1 = tf.keras.layers.Dense(self.hidden_dim, activation='relu', kernel_initializer='he_uniform')
        self.dense_2 = tf.keras.layers.Dense(self.embedding_dim)

    @tf.function
    def call(self, inputs):
        output = self.dense_1(inputs)
        output = self.dense_2(output)
        return output


class Encoder(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, hidden_dim, dropout_rate, head_num):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.head_num = head_num
        self.attention = MultiHeadAttention(self.embedding_dim, self.head_num)
        self.feed_forward = PositionWiseFeedForwardLayer(self.hidden_dim, self.embedding_dim)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(self.dropout_rate)

    @tf.function
    def call(self, inputs, training, mask):
        attention_outputs = self.attention(inputs, mask)
        outputs_1 = self.dropout_1(attention_outputs, training=training)
        outputs_1 = self.layer_norm_1(inputs + outputs_1)
        ffn_output = self.feed_forward(outputs_1)
        outputs_2 = self.dropout_2(ffn_output, training=training)
        outputs_2 = self.layer_norm_2(outputs_1 + outputs_2)
        return outputs_2


class Transformer(tf.keras.Model, ABC):
    def __init__(self, configs, num_classes, vocab_size, embedding_matrix=None):
        super(Transformer, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = configs.hidden_dim
        self.dropout_rate = configs.dropout_rate
        self.encoder_num = configs.encoder_num
        self.head_num = configs.head_num
        self.sequence_length = configs.max_sequence_length
        self.embedding_dim = configs.embedding_dim

        if configs.embedding_method == 'random':
            self.embedding = tf.keras.layers.Embedding(vocab_size + 1, self.embedding_dim, mask_zero=True)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size + 1, self.embedding_dim, weights=[embedding_matrix], trainable=False)

        self.positional_encoder = PositionalEncoding(self.embedding_dim, self.sequence_length)
        self.encoders = [Encoder(self.embedding_dim, self.hidden_dim, self.dropout_rate, self.head_num) for _ in range(self.encoder_num)]
        self.dense = tf.keras.layers.Dense(self.num_classes, activation='softmax')
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.avg_pool = tf.keras.layers.GlobalAvgPool1D()

    @tf.function
    def call(self, inputs, training=None):
        mask = create_padding_mask(inputs)
        embedding_inputs = self.embedding(inputs)
        output = self.positional_encoder(embedding_inputs)
        output = self.dropout(output, training=training)
        for encoder in self.encoders:
            output = encoder(output, training, mask)
        output = self.avg_pool(output)
        output = self.dense(output)
        return output
