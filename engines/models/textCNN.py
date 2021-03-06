from abc import ABC
import tensorflow as tf
from tensorflow.keras import Model


class TextCNN(Model, ABC):
    def __init__(self, configs, num_classes, vocab_size, embedding_matrix=None):
        super(TextCNN, self).__init__()
        self.sequence_length = configs.max_sequence_length
        self.num_filters = configs.num_filters
        self.num_classes = num_classes
        self.embedding_dim = configs.embedding_dim
        self.vocab_size = vocab_size
        self.embedding_matrix = embedding_matrix
        self.dropout_rate = configs.dropout_rate
        self.use_attention = configs.use_attention
        if configs.embedding_method == "random":
            print("ok")
            self.embedding_layer = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim,
                                                             mask_zero=True)
        else:
            self.embedding_layer = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim,
                                                             weights=[self.embedding_matrix], trainable=False)

        if self.use_attention:
            self.attention_size = configs.attention_size
            self.attention_W = tf.keras.layers.Dense(self.attention_size, activation='tanh', use_bias=False)
            self.attention_v = tf.Variable(tf.zeros([1, self.attention_size]))

        self.conv1 = tf.keras.layers.Conv2D(filters=self.num_filters, kernel_size=[2, self.embedding_dim], strides=(1, 1),
                                            padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=[self.sequence_length, 1])
        self.conv2 = tf.keras.layers.Conv2D(filters=self.num_filters, kernel_size=[3, self.embedding_dim], strides=(1, 1),
                                            padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=[self.sequence_length, 1])
        self.conv3 = tf.keras.layers.Conv2D(filters=self.num_filters, kernel_size=[4, self.embedding_dim], strides=(1, 1),
                                            padding='same', activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=[self.sequence_length, 1])
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate, name='dropout')
        self.dense = tf.keras.layers.Dense(self.num_classes, activation='softmax',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.2),
                                           bias_regularizer=tf.keras.regularizers.l2(0.2), name='dense')
        self.flatten = tf.keras.layers.Flatten(data_format='channels_last', name='flatten')
        self.softmax = tf.keras.layers.Softmax()

    @tf.function
    def call(self, inputs, training=None):
        print("inputs1", inputs)
        inputs = self.embedding_layer(inputs)
        print("inputs2", inputs)
        if self.use_attention:
            output = self.attention_W(inputs)
            # print("output1", output)
            output = tf.matmul(output, self.attention_v, transpose_b=True)
            # print("output2", output)
            alpha = tf.nn.softmax(output, axis=1)
            # print("alpha", alpha)
            inputs = alpha * inputs
            # print("result", inputs)

        inputs = tf.expand_dims(inputs, -1)
        pooled_output = []
        con1 = self.conv1(inputs)
        pool1 = self.pool1(con1)
        pooled_output.append(pool1)

        con2 = self.conv2(inputs)
        pool2 = self.pool2(con2)
        pooled_output.append(pool2)

        con3 = self.conv3(inputs)
        pool3 = self.pool3(con3)
        pooled_output.append(pool3)

        concat_outputs = tf.keras.layers.concatenate(pooled_output, axis=-1, name='concatenate')
        flatten_outputs = self.flatten(concat_outputs)
        drop_outputs = self.dropout(flatten_outputs, training)
        outputs = self.dense(drop_outputs)
        # softmax_outputs = self.softmax(outputs)
        # print("12", outputs)
        return outputs
