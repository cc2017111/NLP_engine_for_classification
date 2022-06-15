import numpy as np
import tensorflow as tf
from utils.focal_loss import FocalLoss

class Type(object):
    @classmethod
    def str(cls):
        raise NotImplementedError

class LossType(Type):
    SOFTMAX_CROSS_ENTROPY = "SoftmaxCrossEntropy"
    SOFTMAX_FOCAL_CROSS_ENTROPY = "SoftmaxFocalCrossEntropy"
    SIGMOID_FOCAL_CROSS_ENTROPY = "SigmoidFocalCrossEntropy"
    BCE_WITH_LOGITS = "BCEWithLogitsLoss"

    @classmethod
    def str(cls):
        return ",".join([cls.SOFTMAX_CROSS_ENTROPY, cls.SOFTMAX_FOCAL_CROSS_ENTROPY,
                         cls.SIGMOID_FOCAL_CROSS_ENTROPY, cls.BCE_WITH_LOGITS])


class ClassificationLoss:
    def __init__(self, loss_type=LossType.SOFTMAX_CROSS_ENTROPY, use_hierar=False, is_multi=False):
        super(ClassificationLoss, self).__init__()
        self.loss_type = loss_type
        self.use_hierar = use_hierar
        self.is_multi = is_multi
        if self.loss_type == LossType.SOFTMAX_CROSS_ENTROPY:
            self.criterion = tf.keras.losses.CategoricalCrossentropy()
        elif self.loss_type == LossType.SOFTMAX_FOCAL_CROSS_ENTROPY:
            self.criterion = FocalLoss(is_alpha=True, alpha=0.25, gamma=2.0, epsilon=1e-9)
        elif self.loss_type == LossType.SIGMOID_FOCAL_CROSS_ENTROPY:
            self.criterion = FocalLoss(is_alpha=False, alpha=0.25, gamma=2.0, epsilon=1e-9)
        elif self.loss_type == LossType.BCE_WITH_LOGITS:
            self.criterion = tf.keras.losses.BinaryCrossentropy()
        else:
            raise TypeError(
                "Unsupported loss type: %s. Supported loss type is: %s" % (loss_type, LossType.str())
            )

    def call(self, logits, targets, **argv):
        if self.use_hierar:
            hierar_penalty, hierar_paras, hierar_relations = argv[0:3]
            assert self.loss_type in [LossType.BCE_WITH_LOGITS, LossType.SIGMOID_FOCAL_CROSS_ENTROPY]

            return self.criterion.call(logits, targets) + hierar_penalty * self.cal_recursive_regularize(hierar_paras, hierar_relations)
        else:
            if self.is_multi:
                assert self.loss_type in [LossType.BCE_WITH_LOGITS, LossType.SIGMOID_FOCAL_CROSS_ENTROPY]
            return self.criterion.call(logits, targets)

    def cal_recursive_regularize(self, paras, hierar_relations):
        recursive_loss = 0.0
        for i in range(len(paras)):
            if i not in hierar_relations:
                continue
            children_ids = hierar_relations[i]
            if not children_ids:
                continue
            children_paras = tf.gather(paras, axis=0, indices=children_ids).numpy()
            parent_para = tf.gather(paras, axis=0, indices=i).numpy()
            parent_para = parent_para.repeat(children_ids.size()[0], 1)
            print(parent_para)
            diff_paras = parent_para - children_paras
            diff_paras = np.reshape(diff_paras, (diff_paras.size()[0], -1))
            print(diff_paras)
            recursive_loss += 1.0 / 2 * tf.norm(diff_paras, ord='euclidean') ** 2
        return recursive_loss
