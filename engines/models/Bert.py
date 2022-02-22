import tensorflow as tf
from tensorflow.keras import Model
from transformers import BertConfig, TFBertForSequenceClassification


class Bert_model(Model):
    def __init__(self, bert_path, num_classes):
        super(Bert_model, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)
        self.bert_layers = TFBertForSequenceClassification.from_pretrained(bert_path, num_labels=num_classes)
        self.bert_layers.trainable = True
        self.softmax = tf.keras.layers.Softmax()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 512], dtype=tf.int32, name='input_ids'),
                                  tf.TensorSpec(shape=[None, 512], dtype=tf.int32, name='input_mask'),
                                  tf.TensorSpec(shape=[None, 512], dtype=tf.int32, name='token_ids')])
    def call(self, input_ids, input_mask, token_ids):
        embedding_outputs = self.bert_layers(inputs=input_ids, attention_mask=input_mask, token_type_ids=token_ids)
        softmax_outputs = self.softmax(embedding_outputs)
        softmax_outputs = tf.squeeze(softmax_outputs, axis=0)
        return softmax_outputs
