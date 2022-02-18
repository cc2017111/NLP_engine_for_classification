import math
import time
import warnings
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from engines.models.Bert import Bert_model
from engines.models.textCNN import TextCNN
from engines.utils.metrics import metrics


warnings.filterwarnings("ignore")
pretrain_model_name = "bert-base-chinese"
MODEL_PATH = "./bert-base-chinese"


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

    if configs.use_bert:
        X_train, y_train, att_mask_train, token_type_id_train, X_val, y_val, att_mask_val, token_type_id_val = dataManager.get_training_set()
    else:
        X_train, y_train, X_val, y_val = dataManager.get_training_set()
        att_mask_train, token_type_id_train = np.array([]), np.array([])
        att_mask_val, token_type_id_val = np.array([]), np.array([])

    model = TextCNN(configs=configs, num_classes=num_classes, vocab_size=vocab_size)
    checkpoints = tf.train.Checkpoint(model=model)
    checkpoints_manager = tf.train.CheckpointManager(checkpoint=checkpoints, directory=checkpoint_dir,
                                                     max_to_keep=max_to_keep, checkpoint_name=checkpoint_name)
    num_train_iteration = int(math.ceil(1.0 * len(X_train) / batch_size))
    num_val_iteration = int(math.ceil(1.0 * len(X_val) / batch_size))
    logger.info(('+' * 20) + 'training start' + ('+' * 20))

    for i in range(epoch):
        start_time = time.time()
        sh_index = np.arange(len(X_train))
        np.random.shuffle(sh_index)
        X_train = X_train[sh_index]
        y_train = y_train[sh_index]
        if configs.use_bert:
            att_mask_train = att_mask_train[sh_index]
            token_type_id_train = token_type_id_train[sh_index]

        sh_index_val = np.arange(len(X_val))
        np.random.shuffle(sh_index_val)
        X_val = X_val[sh_index_val]
        y_val = y_val[sh_index_val]
        if configs.use_bert:
            att_mask_val = att_mask_val[sh_index_val]
            token_type_id_val = token_type_id_val[sh_index_val]

        logger.info('epoch: {}/{}'.format(i + 1, epoch))
        train_results = {}
        train_loss_values = 0
        for measure in configs.measuring_metrics:
            train_results[measure] = 0
        for iteration in tqdm(range(num_train_iteration)):
            if configs.use_bert:
                X_train_batch, y_train_batch, \
                att_mask_train_batch, token_type_id_train_batch = dataManager.next_batch(X_train, y_train, att_mask_train,
                                                                                         token_type_id_train,
                                                                                         start_index=iteration * batch_size)
            else:
                X_train_batch, y_train_batch = dataManager.next_batch(X_train, y_train, start_index=iteration*batch_size)

            with tf.GradientTape() as tape:
                outputs = model.call(inputs=X_train_batch)
                y_true = tf.one_hot(y_train_batch, depth=num_classes)
                losses = tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=outputs, from_logits=False)
                loss = tf.reduce_mean(losses)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            measures = metrics(y_train_batch, outputs, configs)
            for k, v in measures.items():
                train_results[k] += v
            train_loss_values += loss

            if iteration % configs.print_per_batch == 0 and iteration != 0:
                res_str = ''
                for k, v in measures.items():
                    train_results[k] /= configs.print_per_batch
                    res_str += (k + ':%.3f' % v)
                logger.info('training batch: %5d, loss:%.5f, %s' % (iteration, train_loss_values/configs.print_per_batch, res_str))
                train_loss_values = 0
                for measures in configs.measuring_metrics:
                    train_results[measures] = 0

        # validation
        logger.info('start evaluate engines...')
        loss_values = []
        val_results = {}
        for measure in configs.measuring_metrics:
            val_results[measure] = 0
        for iteration in tqdm(range(num_val_iteration)):
            if configs.use_bert:
                x_val_batch, y_val_batch, att_mask_val_batch, token_type_id_val_batch = dataManager.next_batch(X_val, y_val, att_mask_val, token_type_id_val, start_index=iteration * batch_size)
                outputs_val = model.call(inputs=x_val_batch)
                y_true = tf.one_hot(y_val_batch, depth=num_classes)
                val_losses = tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=outputs_val, from_logits=False)
                val_loss = tf.reduce_mean(val_losses)
                measures = metrics(y_val_batch, outputs_val, configs)

                for k, v in measures.items():
                    val_results[k] += v
                loss_values.append(val_loss)

            time_span = (time.time() - start_time) / 60
            val_res_str = ''
            dev_accuracy_avg = 0
            for k, v in val_results.items():
                val_results[k] /= num_val_iteration
                val_res_str += (k + ':%.3f' % val_results[k])
                if k == 'accuracy':
                    dev_accuracy_avg = val_results[k]

            logger.info('time consumption: %.2f(min), %s' % (time_span, val_res_str))
            print('best acc:', np.array(dev_accuracy_avg).mean())

            if np.array(dev_accuracy_avg).mean() > best_acc:
                unprocess = 0
                best_acc = np.array(dev_accuracy_avg).mean()
                best_at_epoch = i + 1
                checkpoints_manager.save()
                tf.saved_model.save(model, configs.pb_model_sava_dir)
                logger.info('saved the new best model with acc: %.3f' % best_acc)
            else:
                unprocess += 1

            if configs.is_early_stop:
                if unprocess >= configs.patient:
                    logger.info('early stopped, no process obtained with {} epoch'. format(configs.patient))
                    logger.info('overall best acc is {} at {} epoch'.format(best_acc, best_at_epoch))
                    logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
                    return
    logger.info('overall best acc is {} at {} epoch'.format(best_acc, best_at_epoch))
    logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
