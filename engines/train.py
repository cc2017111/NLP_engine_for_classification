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
from utils.metrics import metrics

base_path = Path(__file__).resolve().parent.parent
warnings.filterwarnings("ignore")
pretrain_model_name = "bert-base-chinese"
MODEL_PATH = "./bert-base-chinese"
strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1", "GPU:2"])
stamp = datetime.datetime.now().strftime("Y%m-%d-%H-%M-%S")
tensorboard_log_dir = os.path.join(base_path, "tensorboard_logs/"+stamp)


def train(configs, dataManager, logger):
    vocab_size = dataManager.max_token_num
    num_classes = dataManager.max_label_num
    learning_rate = configs.learning_rate
    max_to_keep = configs.max_to_keep
    checkpoint_dir = configs.checkpoint_dir
    checkpoint_name = configs.checkpoint_name
    best_acc = 0.0
    best_at_epoch = 0
    unprocess = 0
    very_start_time = time.time()
    epoch = configs.epoch
    batch_size = configs.batch_size
    global_batch_size = batch_size * strategy.num_replicas_in_sync

    train_dataset, valid_dataset = dataManager.get_training_dataset()
    train_dataset = train_dataset.batch(global_batch_size).prefetch(global_batch_size * 2)
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

    valid_dataset = valid_dataset.batch(global_batch_size).prefetch(global_batch_size * 2)
    valid_dist_dataset = strategy.experimental_distribute_dataset(valid_dataset)

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
            loss_fn = ClassificationLoss(loss_type=loss_type, use_hierar=False, is_multi=False)
            
        elif configs.model == "HMCN":
            tree_label_file = str(base_path) + '/data/vocabs/tree_label_file'
            hierarchical_class, hierar_relations = get_hierar_relations(tree_label_file, dataManager.label2id)
            model = HMCN(configs=configs, bert_path=configs.bert_pretrain_path, hierarchical_class=hierarchical_class, hierar_relations=hierar_relations)
            loss_type = LossType.SIGMOID_FOCAL_CROSS_ENTROPY
            loss_fn = ClassificationLoss(loss_type=loss_type, use_hierar=True, is_multi=False)
        else:
            model = Transformer(configs=configs, num_classes=num_classes, vocab_size=vocab_size)
            loss_type = LossType.SOFTMAX_FOCAL_CROSS_ENTROPY
            loss_fn = ClassificationLoss(loss_type=loss_type, use_hierar=False, is_multi=False)

        if configs.optimizer == "Adagrad":
            optimizer = tf.keras.optimizers.Adagrad(learning_rate)
        elif configs.optimizer == "Adadelta":
            optimizer = tf.keras.optimizers.Adadelta(learning_rate)
        elif configs.optimizer == "RMSProp":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate)
        elif configs.optimizer == "SGD":
            optimizer = tf.keras.optimizers.SGD(learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate)

        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        checkpoints = tf.train.Checkpoint(model=model)
        checkpoints_manager = tf.train.CheckpointManager(checkpoint=checkpoints, directory=checkpoint_dir,
                                                         max_to_keep=max_to_keep, checkpoint_name=checkpoint_name)

    @tf.function
    def train_step(inputs):
        with tf.GradientTape() as tape:
            if configs.model == "BERT":
                X_train_batch, y_train_batch, att_mask_train_batch, token_type_ids_train_batch = inputs
                # print(X_train_batch, y_train_batch, att_mask_train_batch, token_type_ids_train_batch)
                outputs = model.call(input_ids=X_train_batch, input_mask=att_mask_train_batch, token_ids=token_type_ids_train_batch)
            elif configs.model == 'HMCN':
                X_train_batch, y_train_batch, att_mask_train_batch, token_type_ids_train_batch = inputs
                global_layer_output, local_layer_outputs, infer_outputs = model.call(input_ids=X_train_batch, input_mask=att_mask_train_batch, token_ids=token_type_ids_train_batch)
            else:
                X_train_batch, y_train_batch = inputs
                outputs = model.call(inputs=X_train_batch)
            y_true = tf.one_hot(y_train_batch, depth=num_classes)
            if configs.model == "HMCN":
                losses_train = loss_fn.call(targets=y_true, logits=global_layer_output, hierar_penalty=configs.hierar_penalty, hierar_paras=model.linear.weights, hierar_relations=hierar_relations)
                losses_train += loss_fn.call(targets=y_true, logits=local_layer_outputs, hierar_penalty=configs.hierar_penalty, hierar_paras=model.linear.weights, hierar_relations=hierar_relations)
            else:
                losses_train = loss_fn.call(targets=y_true, logits=outputs)
            loss_train = tf.reduce_mean(losses_train)
            train_loss.update_state(loss_train)

        gradients = tape.gradient(loss_train, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_accuracy.update_state(y_train_batch, outputs)
        return loss_train

    @tf.function
    def test_step(inputs):
        if configs.model == "BERT":
            X_val_batch, y_val_batch, att_mask_val_batch, token_type_ids_val_batch = inputs
            outputs = model.call(input_ids=X_val_batch, input_mask=att_mask_val_batch, token_ids=token_type_ids_val_batch)
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
        return loss_val

    @tf.function
    def distributed_train_step(inputs):
        per_replica_losses = strategy.run(train_step, args=(inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tf.function
    def distributed_test_step(inputs):
        return strategy.run(test_step, args=(inputs,))

    summary_writer = tf.summary.create_file_writer(tensorboard_log_dir)
    tf.summary.trace_on(graph=True, profiler=True)

    logger.info(('+' * 20) + 'training start' + ('+' * 20))
    for i in range(epoch):
        logger.info('epoch: {}/{}'.format(i + 1, epoch))
        start_time = time.time()
        loss_train = 0.0
        acc_train = 0.0
        num_train_batches = 0
        num_val_batches = 0
        train_accuracy.reset_state()
        test_accuracy.reset_state()
        test_loss.reset_state()
        train_loss.reset_state()

        checkpoint_path = checkpoints_manager.restore_or_initialize()
        if checkpoint_path is not None:
            logger.info("restore checkpoint at {}".format(checkpoint_path))
        for X in iter(train_dist_dataset):
            # print(X)
            _ = distributed_train_step(X)
            num_train_batches += 1
            loss_train += train_loss.result()
            acc_train += train_accuracy.result()
            if num_train_batches % configs.print_per_batch == 0 and num_train_batches != 0:
                logger.info(
                    'training batch: %5d, loss:%.5f, acc:%.5f' % (num_train_batches, loss_train / configs.print_per_batch, acc_train / configs.print_per_batch))
                loss_train = 0.0
                acc_train = 0.0

            if num_train_batches % configs.save_interval_updates == 0 and num_train_batches != 0:
                logger.info(
                    'training batch: %5d, save ckeckpoints: %5d' % (num_train_batches, num_train_batches / configs.save_interval_updates)
                )
                checkpoints_manager.save()

        with summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        # validation
        logger.info('start evaluate engines...')
        loss_val = 0.0
        acc_val = 0.0

        for X_val in iter(valid_dist_dataset):
            _ = distributed_test_step(X_val)
            num_val_batches += 1

            loss_val += test_loss.result()
            acc_val += test_accuracy.result()
            if num_val_batches % configs.print_per_batch == 0 and num_val_batches != 0:
                logger.info(
                    'validating batch: %5d, loss:%.5f, acc:%.5f' % (num_val_batches, loss_val / configs.print_per_batch, acc_val / configs.print_per_batch))
                loss_val = 0.0
                acc_val = 0.0

        time_span = (time.time() - start_time) / 60
        logger.info('time consumption: %.2f(min)' % time_span)
        val_acc = test_accuracy.result()

        with summary_writer.as_default():
            tf.summary.scalar('test_loss', test_loss.result(), step=epoch)
            tf.summary.scalar('test_accuracy', test_accuracy.result(), step=epoch)

        if val_acc > best_acc:
            unprocess = 0
            best_acc = val_acc
            best_at_epoch = i + 1
            checkpoints_manager.save()
            tf.saved_model.save(model, configs.pb_model_sava_dir)
            logger.info('saved the new best model with acc: %.3f' % best_acc)
        else:
            unprocess += 1
        print('best acc:', best_acc)

        if configs.is_early_stop:
            if unprocess >= configs.patient:
                logger.info('early stopped, no process obtained with {} epoch'. format(configs.patient))
                logger.info('overall best acc is {} at {} epoch'.format(best_acc, best_at_epoch))
                logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
                return
    with summary_writer.as_default():
        tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=tensorboard_log_dir)

    logger.info('overall best acc is {} at {} epoch'.format(best_acc, best_at_epoch))
    logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
