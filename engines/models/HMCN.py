import tensorflow as tf
from tensorflow.keras import Model
from transformers import BertConfig, TFBertModel


class HMCN(Model):
    def __init__(self, configs, bert_path, hierarchical_class, hierar_relations):
        super(HMCN, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)
        self.hierarchical_depth = configs.hierarchical_depth
        self.global2local = configs.global2local
        self.hierarchical_class = hierarchical_class
        self.hierar_relations = hierar_relations
        self.dropout_rate = configs.dropout_rate
        self.bert_layers = TFBertModel.from_pretrained(bert_path)
        self.bert_layers.trainable = True
        self.softmax = tf.keras.layers.Softmax()
        self.local_layers = []
        self.global_layers = []
        for i in range(len(self.hierarchical_depth)):
            temp_global_model = tf.keras.Sequential([tf.keras.layers.Dense(self.hierarchical_depth[i]),
                                                    tf.keras.layers.ReLU(),
                                                    tf.keras.layers.BatchNormalization(),
                                                    tf.keras.layers.Dropout(self.dropout_rate)])
            self.global_layers.append(temp_global_model)
            temp_local_model = tf.keras.Sequential([tf.keras.layers.Dense(self.global2local[i]),
                                                    tf.keras.layers.ReLU(),
                                                    tf.keras.layers.BatchNormalization(),
                                                    tf.keras.layers.Dense(self.hierarchical_class[i])])
            self.local_layers.append(temp_local_model)
        self.linear = tf.keras.layers.Dense(self.hierarchical_class[-1])
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 512], dtype=tf.int32, name='input_ids'),
                                  tf.TensorSpec(shape=[None, 512], dtype=tf.int32, name='input_mask'),
                                  tf.TensorSpec(shape=[None, 512], dtype=tf.int32, name='token_ids')])
    def call(self, input_ids, input_mask, token_ids):
        embedding_outputs = self.bert_layers(inputs=input_ids, attention_mask=input_mask, token_type_ids=token_ids)
        inputs = embedding_outputs[0]
        global_layer_activation = inputs
        local_layer_outputs = []
        for i, (local_layer, global_layer) in enumerate(zip(self.local_layers, self.global_layers)):
            local_layer_activation = global_layer(global_layer_activation)
            local_layer_outputs.append(local_layer(local_layer_activation))
            if i < len(self.global_layers) - 1:
                global_layer_activation = tf.concat([local_layer_activation, inputs], axis=1)
            else:
                global_layer_activation = local_layer_activation
        global_layer_output = self.linear(global_layer_activation)
        local_layer_outputs = tf.concat(local_layer_outputs, axis=1)
        return global_layer_output, local_layer_outputs, 0.5 * global_layer_output + 0.5 * local_layer_outputs

    def cal_recursive_regularize(self, params):

        pass