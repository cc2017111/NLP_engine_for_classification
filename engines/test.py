import math
import os
import time
import warnings
import datetime
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
from models.Bert import Bert_model
from models.textCNN import TextCNN
from models.TextRNN import TextRNN
from models.TextRCNN import TextRCNN
from models.Transformer import Transformer
from models.FastText import FastText
from models.HMCN import HMCN
from utils.get_hierar_relations import get_hierar_relations
from utils.ClassificationLoss import ClassificationLoss, LossType
from utils.plot import plot_confusion_matrix
from utils.metrics import metrics


warnings.filterwarnings("ignore")
pretrain_model_name = "bert-base-chinese"
MODEL_PATH = "./bert-base-chinese"
strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1", "GPU:2"])
base_path = Path(__file__).resolve().parent.parent


def test(configs, dataManager, logger):
    vocab_size = dataManager.max_token_num
    num_classes = dataManager.max_label_num
    max_to_keep = configs.max_to_keep
    checkpoint_dir = configs.checkpoint_dir
    checkpoint_name = configs.checkpoint_name
    batch_size = configs.batch_size
    global_batch_size = batch_size * strategy.num_replicas_in_sync

    test_set = dataManager.get_testing_dataset()

    test_dataset = test_set.batch(global_batch_size).prefetch(global_batch_size * 2)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)
    with strategy.scope():
        if configs.model == "BERT":
            model = Bert_model(configs=configs, bert_path=configs.bert_pretrain_path, num_classes=num_classes)
            loss_type = LossType.SOFTMAX_FOCAL_CROSS_ENTROPY
            loss_fn = ClassificationLoss(loss_type=loss_type, use_hierar=False, is_multi=False)

        elif configs.model == "TextCNN":
            model = TextCNN(configs=configs, num_classes=num_classes, vocab_size=vocab_size)
            loss_type = LossType.SOFTMAX_CROSS_ENTROPY
            loss_fn = ClassificationLoss(loss_type=loss_type, use_hierar=False, is_multi=False)

        elif configs.model == "TextRNN":
            model = TextRNN(configs=configs, num_classes=num_classes, vocab_size=vocab_size)
            loss_type = LossType.SOFTMAX_CROSS_ENTROPY
            loss_fn = ClassificationLoss(loss_type=loss_type, use_hierar=False, is_multi=False)

        elif configs.model == "TextRCNN":
            model = TextRCNN(configs=configs, num_classes=num_classes, vocab_size=vocab_size)
            loss_type = LossType.SOFTMAX_CROSS_ENTROPY
            loss_fn = ClassificationLoss(loss_type=loss_type, use_hierar=False, is_multi=False)

        elif configs.model == "FastText":
            model = FastText(configs=configs, num_classes=num_classes, vocab_size=vocab_size)
            loss_type = LossType.SOFTMAX_CROSS_ENTROPY
            loss_fn = ClassificationLoss(loss_type=loss_type, use_hierar=True, is_multi=False)

        elif configs.model == "HMCN":
            tree_label_file = str(base_path) + '/data/vocabs/tree_label_file'
            hierarchical_class, hierar_relations = get_hierar_relations(tree_label_file, dataManager.label2id)
            model = HMCN(configs=configs, bert_path=configs.bert_pretrain_path, hierarchical_class=hierarchical_class,
                         hierar_relations=hierar_relations)
            loss_type = LossType.SIGMOID_FOCAL_CROSS_ENTROPY
            loss_fn = ClassificationLoss(loss_type=loss_type, use_hierar=True, is_multi=False)
        else:
            model = Transformer(configs=configs, num_classes=num_classes, vocab_size=vocab_size)
            loss_type = LossType.SOFTMAX_FOCAL_CROSS_ENTROPY
            loss_fn = ClassificationLoss(loss_type=loss_type, use_hierar=True, is_multi=False)

        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        checkpoints = tf.train.Checkpoint(model=model)
        checkpoints_manager = tf.train.CheckpointManager(checkpoint=checkpoints, directory=checkpoint_dir,
                                                         max_to_keep=max_to_keep, checkpoint_name=checkpoint_name)

    @tf.function
    def test_step(inputs):
        if configs.model == "BERT":
            X_val_batch, y_val_batch, att_mask_val_batch, token_type_ids_val_batch = inputs
            outputs = model.call(input_ids=X_val_batch, input_mask=att_mask_val_batch,
                                 token_ids=token_type_ids_val_batch)
        elif configs.model == 'HMCN':
            X_val_batch, y_val_batch, att_mask_val_batch, token_type_ids_val_batch = inputs
            global_layer_output_val, local_layer_outputs_val, infer_outputs_val = model.call(input_ids=X_val_batch,
                                                                                             input_mask=att_mask_val_batch,
                                                                                             token_ids=token_type_ids_val_batch)
        else:
            X_val_batch, y_val_batch = inputs
            outputs = model.call(inputs=X_val_batch)

        y_true = tf.one_hot(y_val_batch, depth=num_classes)
        if configs.model == "HMCN":
            losses_val = loss_fn.call(targets=y_true, logits=global_layer_output_val,
                                      hierar_penalty=configs.hierar_penalty, hierar_paras=model.linear.weights,
                                      hierar_relations=hierar_relations)
            losses_val += loss_fn.call(targets=y_true, logits=local_layer_outputs_val,
                                       hierar_penalty=configs.hierar_penalty, hierar_paras=model.linear.weights,
                                       hierar_relations=hierar_relations)
        else:
            losses_val = loss_fn.call(targets=y_true, logits=outputs)
        loss_val = tf.reduce_mean(losses_val)
        test_loss.update_state(loss_val)
        test_accuracy.update_state(y_val_batch, outputs)
        return loss_val, outputs

    @tf.function
    def distributed_test_step(inputs):
        return strategy.run(test_step, args=(inputs,))

    logger.info(('+' * 20) + 'testing start' + ('+' * 20))
    total_acc = 0.0
    loss_val = 0.0
    acc_val = 0.0
    start_time = time.time()
    num_test_batches = 0
    y_true = []
    y_pred = []
    checkpoint_path = checkpoints_manager.restore_or_initialize()
    if checkpoint_path is not None:
        logger.info("restore checkpoint at {}".format(checkpoint_path))
    for X in iter(test_dist_dataset):
        X_val_batch, y_val_batch, att_mask_val_batch, token_type_ids_val_batch = X
        _, logits = distributed_test_step(X)
        logits = list(logits.values)
        logits = tf.concat(logits, axis=0)
        logits = tf.argmax(logits, axis=1).numpy()
        label_true = np.concatenate(y_val_batch.values, axis=0)

        y_true.extend(label_true)
        y_pred.extend(logits)
        num_test_batches += 1

        total_acc += test_loss.result()
        loss_val += test_loss.result()
        acc_val += test_accuracy.result()
        if num_test_batches % configs.print_per_batch == 0 and num_test_batches != 0:
            logger.info(
                'validating batch: %5d, loss:%.5f, acc:%.5f' % (
                num_test_batches, loss_val / configs.print_per_batch, acc_val / configs.print_per_batch))
            loss_val = 0.0
            acc_val = 0.0

    def load_labels_level_n(n):
        label2id_file = str(base_path) + '/' + configs.vocabs_dir + '/label2id_level' + str(n) + '_2m'
        with open(label2id_file, mode='r', encoding='utf-8') as file:
            rows = file.readlines()
            label2id = {}
            id2label = {}
            for row in rows:
                label = row.split('\t')[0]
                id = row.split('\t')[1].strip()
                label2id[label] = id
                id2label[id] = label

        return label2id, id2label

    def get_level_1_label(y):
        label2id_level1, id2label_level1 = load_labels_level_n(1)
        label2id_level3, id2label_level3 = load_labels_level_n(3)
        temp = []
        for i in y:
            temp.append(id2label_level3[str(i)][0])
        res = [int(label2id_level1[i]) for i in temp]
        return res

    y_true = get_level_1_label(y_true)
    y_pred = get_level_1_label(y_pred)
    level1_class_num = 8
    label2id_level1, id2label_level1 = load_labels_level_n(1)
    label_list = list(label2id_level1.keys())
    print(label_list)
    plot_confusion_matrix(y_true, y_pred, level1_class_num, label_list=label_list)
    time_span = (time.time() - start_time) / 60
    logger.info('total loss: %2f' % (total_acc / num_test_batches))
    logger.info('time consumption: %.2f(min)' % time_span)
    # test_accuracy.reset_state()
    # test_loss.reset_state()

