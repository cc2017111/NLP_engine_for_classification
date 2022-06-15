import os
import random
import jieba
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from pathlib import Path
from transformers import BertTokenizer
from collections import Counter
from utils.io_functions import read_csv
from utils.w2v_utils import Word2VecUtils
from utils.clean_data import filter_word, filter_char
base_path = Path(__file__).resolve().parent.parent


class BertDataManager:
    def __init__(self, configs, logger):
        self.configs = configs
        self.logger = logger

        self.train_file = str(base_path) + '/' + configs.datasets_fold + '/' + configs.train_file
        if configs.dev_file is not None:
            self.dev_file = str(base_path) + '/' + configs.datasets_fold + '/' + configs.dev_file
        else:
            self.dev_file = None

        self.PADDING = '[PAD]'
        self.batch_size = configs.batch_size
        self.max_sequence_length = configs.max_sequence_length
        self.vocabs_dir = configs.vocabs_dir
        self.label2id_file = str(base_path) + '/' + configs.vocabs_dir + '/label2id'
        self.label2id, self.id2label = self.load_labels()
        self.tfrecords_file_path = str(base_path) + '/' + configs.tfrecords_file_path

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.max_token_num = len(self.tokenizer.get_vocab())
        self.max_label_num = len(self.label2id)

    def load_labels(self):
        if not os.path.isfile(self.label2id_file):
            self.logger.info("label2id file does not exists, creating...")
            return self.build_labels()
        with open(self.label2id_file, mode='r', encoding='utf-8') as file:
            rows = file.readlines()
            label2id = {}
            id2label = {}
            for row in rows:
                label = row.split('\t')[0]
                id = row.split('\t')[1].strip()
                label2id[label] = id
                id2label[id] = label

        return label2id, id2label

    def build_labels(self):
        df_train = read_csv(self.train_file, names=['sentence', 'label'], delimiter=',')
        labels = list(set(df_train['label'][df_train['label'].notnull()]))
        for i in range(len(labels)):
            labels[i] = labels[i]
        labels = list(set(labels))
        id2label = dict(zip(range(0, len(labels)), labels))
        label2id = dict(zip(labels, range(0, len(labels))))
        with open(self.label2id_file, mode='w', encoding='utf-8') as outfile:
            for idx in id2label:
                outfile.write(str(id2label[idx]) + '\t' + str(idx) + '\n')

        return label2id, id2label

    def get_training_set(self, ratio=0.8):
        df_train = read_csv(self.train_file, names=['sentence', 'label'], delimiter=',')
        X, y, att_mask, token_type_ids = self.prepare(df_train)

        num_samples = len(X)
        if self.dev_file is not None:
            X_train = X
            y_train = y
            att_mask_train = att_mask
            token_type_ids_train = token_type_ids
            X_val, y_val, att_mask_val, token_type_ids_val = self.get_valid_set()
        else:
            X_train = X[:int(num_samples * ratio)]
            y_train = y[:int(num_samples * ratio)]
            X_val = X[int(num_samples * ratio):]
            y_val = y[int(num_samples * ratio):]
            att_mask_train = att_mask[:int(num_samples * ratio)]
            att_mask_val = att_mask[int(num_samples * ratio):]
            token_type_ids_train = token_type_ids[:int(num_samples * ratio)]
            token_type_ids_val = token_type_ids[int(num_samples * ratio):]
            self.logger.info("validation set does not exist, built...")
        self.logger.info("train set size: {}, validation set size: {}".format(len(X_train), len(X_val)))
        return X_train, y_train, att_mask_train, token_type_ids_train, X_val, y_val, att_mask_val, token_type_ids_val

    def save_tfrecords(self, dataframe, desfile):
        with tf.io.TFRecordWriter(os.path.join(self.tfrecords_file_path, desfile)) as writer:
            self.logger.info("saving tfrecords data...")
            for index, record in tqdm(dataframe.iterrows()):
                sentence = record.sentence.replace(' ', '')
                label = record.label
                if len(sentence) < self.max_sequence_length - 2:
                    tmp_x = self.tokenizer.encode(sentence)
                    tmp_att_mask = [1] * len(tmp_x)
                    tmp_y = self.label2id[label]
                    tmp_x += [0 for _ in range(self.max_sequence_length - len(tmp_x))]
                    tmp_att_mask += [0 for _ in range(self.max_sequence_length - len(tmp_att_mask))]
                    tmp_token_type_ids = self.max_sequence_length * [0]
                else:
                    tmp_x = self.tokenizer.encode(sentence)
                    tmp_x = tmp_x[:self.max_sequence_length]
                    if len(tmp_x) != 512:
                        print(len(tmp_x))
                        print(index)
                        print(record.sentence)
                    assert len(tmp_x) == 512
                    tmp_y = self.label2id[label]
                    tmp_att_mask = [1] * self.max_sequence_length
                    tmp_token_type_ids = [0] * self.max_sequence_length
                features = tf.train.Features(feature={
                    "X": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=tmp_x)),
                    "y": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[int(tmp_y)])),
                    "att_mask": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=tmp_att_mask)),
                    "token_type_ids": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=tmp_token_type_ids)),
                })
                example = tf.train.Example(features=features)
                serialized = example.SerializeToString()
                writer.write(serialized)

    def get_labels(self):
        df_train = read_csv(self.train_file, names=['sentence', 'label'], delimiter=',')
        y = []
        for index, record in tqdm(df_train.iterrows()):
            label = record.label
            tmp_y = self.label2id[label]
            y.append(tmp_y)
        return np.array(y, dtype=np.int32)

    def get_valid_set(self):
        df_val = read_csv(self.dev_file, names=['sentence', 'label'], delimiter=',')
        X_val, y_val, att_mask_val, token_type_ids_val = self.prepare(df_val)
        return X_val, y_val, att_mask_val, token_type_ids_val

    def get_training_dataset(self):
        # X_train, y_train, att_mask_train, token_type_ids_train, X_val, y_val, att_mask_val, token_type_ids_val = self.get_training_set(ratio=0.8)
        #
        # train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, att_mask_train, token_type_ids_train))
        # valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val, att_mask_val, token_type_ids_val))
        # return train_dataset, valid_dataset
        if not os.path.exists(os.path.join(self.tfrecords_file_path, "train.tfrecords")):
            df_train = read_csv(self.train_file, names=['sentence', 'label'], delimiter=',')
            self.save_tfrecords(df_train, "train.tfrecords")
        if not os.path.exists(os.path.join(self.tfrecords_file_path, "test.tfrecords")):
            df_test = read_csv(self.dev_file, names=['sentence', 'label'], delimiter=',')
            self.save_tfrecords(df_test, "test.tfrecords")
        # ex = next(tf.compat.v1.python_io.tf_record_iterator(os.path.join(self.tfrecords_file_path, "train.tfrecords")))
        # print(tf.train.Example.FromString(ex))
        train_dataset = self.load_dataset("train.tfrecords")
        test_dataset = self.load_dataset("test.tfrecords")
        return train_dataset, test_dataset

    def get_testing_dataset(self):
        if not os.path.exists(os.path.join(self.tfrecords_file_path, "test.tfrecords")):
            df_test = read_csv(self.dev_file, names=['sentence', 'label'], delimiter=',')
            self.save_tfrecords(df_test, "test.tfrecords")
        # ex = next(tf.compat.v1.python_io.tf_record_iterator(os.path.join(self.tfrecords_file_path, "train.tfrecords")))
        # print(tf.train.Example.FromString(ex))
        test_dataset = self.load_dataset("test.tfrecords")
        return test_dataset
        # X_val, y_val, att_mask_val, token_type_ids_val = self.get_valid_set()
        # valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val, att_mask_val, token_type_ids_val))
        # return valid_dataset

    def prepare(self, df):
        self.logger.info("loading data...")
        X= []
        y = []
        att_mask = []
        token_type_ids = []
        for index, record in tqdm(df.iterrows()):
            sentence = record.sentence.replace(' ', '')
            label = record.label
            # print(label)
            if len(sentence) < self.max_sequence_length - 2:
                tmp_x = self.tokenizer.encode(sentence)
                tmp_att_mask = [1] * len(tmp_x)
                tmp_y = self.label2id[label]

                tmp_x += [0 for _ in range(self.max_sequence_length - len(tmp_x))]
                tmp_att_mask += [0 for _ in range(self.max_sequence_length - len(tmp_att_mask))]
                tmp_token_type_ids = self.max_sequence_length * [0]
                X.append(tmp_x)
                y.append(tmp_y)
                att_mask.append(tmp_att_mask)
                token_type_ids.append(tmp_token_type_ids)
            else:
                tmp_x = self.tokenizer.encode(sentence)
                tmp_x = tmp_x[:self.max_sequence_length]
                if len(tmp_x) != 512:
                    print(len(tmp_x))
                    print(index)
                    print(record.sentence)
                assert len(tmp_x) == 512
                tmp_y = self.label2id[label]
                tmp_att_mask = [1] * self.max_sequence_length
                tmp_token_type_ids = [0] * self.max_sequence_length
                X.append(tmp_x)
                y.append(tmp_y)
                att_mask.append(tmp_att_mask)
                token_type_ids.append(tmp_token_type_ids)

        return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32), np.array(att_mask, dtype=np.int32), np.array(token_type_ids, dtype=np.int32)

    def next_batch(self, x, y, att_mask, token_type_ids, start_index):
        last_index = start_index + self.batch_size
        x_batch = list(x[start_index:min(last_index, len(x))])
        y_batch = list(y[start_index:min(last_index, len(x))])
        att_mask_batch = list(token_type_ids[start_index:min(last_index, len(x))])
        token_type_ids_batch = list(token_type_ids[start_index:min(last_index, len(x))])
        if last_index > len(x):
            left_size = last_index - len(x)
            for i in range(left_size):
                index = np.random.randint(len(x))
                x_batch.append(x[index])
                y_batch.append(y[index])
                att_mask_batch.append(att_mask[index])
                token_type_ids_batch.append(token_type_ids[index])
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        att_mask_batch = np.array(att_mask_batch)
        token_type_ids_batch = np.array(token_type_ids_batch)

        return x_batch, y_batch, att_mask_batch, token_type_ids_batch

    def map_func(self, example):
        feature_description = {
            'X': tf.io.FixedLenFeature([512], tf.int64),
            'y': tf.io.FixedLenFeature([1], tf.int64),
            'att_mask': tf.io.FixedLenFeature([512], tf.int64),
            'token_type_ids': tf.io.FixedLenFeature([512], tf.int64)
        }
        parsed_example = tf.io.parse_single_example(example, features=feature_description)
        X = tf.cast(parsed_example['X'], tf.int32)
        # tf.print(X)
        y = tf.squeeze(tf.cast(parsed_example['y'], tf.int32))
        # tf.print(y)
        att_mask = tf.cast(parsed_example['att_mask'], tf.int32)
        # tf.print(att_mask)
        token_type_ids = tf.cast(parsed_example['token_type_ids'], tf.int32)
        # tf.print(token_type_ids)
        return X, y, att_mask, token_type_ids

    def load_dataset(self, filepath):
        shuffle_block = 140000
        dataset = tf.data.TFRecordDataset(os.path.join(self.tfrecords_file_path, filepath))
        dataset = dataset.map(map_func=self.map_func, num_parallel_calls=8)
        dataset = dataset.shuffle(shuffle_block)
        return dataset


class DataManager:
    def __init__(self, configs, logger):
        self.configs = configs
        self.logger = logger

        self.UNKNOWN = '[UNK]'
        self.PADDING = '[PAD]'

        self.train_file = str(base_path) + '/' + configs.datasets_fold + '/' + configs.train_file
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        if configs.dev_file is not None:
            self.dev_file = str(base_path) + '/' + configs.datasets_fold + '/' + configs.dev_file
        else:
            self.dev_file = None

        self.batch_size = configs.batch_size
        self.max_sequence_length = configs.max_sequence_length
        self.embedding_dim = configs.embedding_dim
        self.vocabs_dir = configs.vocabs_dir
        self.token_level = configs.token_level
        self.token2id_file = str(base_path) + '/' + configs.vocabs_dir + '/token2id'
        self.label2id_file = str(base_path) + '/' + configs.vocabs_dir + '/label2id'

        self.token2id, self.id2token, self.label2id, self.id2label = self.load_vocab()
        self.max_sequence_length = configs.max_sequence_length
        self.max_token_num = len(self.tokenizer.get_vocab())
        self.max_label_num = len(self.label2id)
        self.word2vec_utils = Word2VecUtils(self.configs, self.logger)
        self.logger.info('dataManager initialized...')

    def load_vocab(self):
        if not os.path.isfile(self.token2id_file):
            self.logger.info('vocab file not exist, building vocab...')
            return self.build_vocab()

        self.logger.info('loading vocab...')
        token2id, id2token = {}, {}
        with open(self.token2id_file, mode='r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                token, token_id = row.split('\t')[0], int(row.split('\t')[1])
                token2id[token] = token_id
                id2token[token_id] = token

        label2id, id2label = {}, {}
        with open(self.label2id_file, mode='r', encoding='utf-8') as infile:
            for row in infile:
                row.strip()
                label, label_id = row.split('\t')[0], int(row.split('\t')[1])
                label2id[label] = label_id
                id2label[label_id] = label

        return token2id, id2token, label2id, id2label

    def build_vocab(self):
        df_train = read_csv(self.train_file, names=['sentence', 'label'], delimiter=',')
        sentences = list(set(df_train['sentence'][df_train['sentence'].notnull()]))
        tokens = []
        if self.token_level == 'word':
            for sentence in tqdm(sentences):
                words = self.word2vec_utils.processing_sentence(sentence, self.word2vec_utils.get_stop_word())
                tokens.extend(words)

            count_dict = Counter(tokens)
            tokens = [k for k, v in count_dict.items() if v > 1 and filter_word(k)]
        else:
            for sentence in tqdm(sentences):
                chars = list(sentence)
                tokens.extend(chars)
            count_dict = Counter(tokens)
            tokens = [k for k, v in count_dict.items() if k != ' ' and filter_char(k)]
        tokens = set(tokens)
        labels = list(set(df_train['label'][df_train['label'].notnull()]))
        token2id = dict(zip(tokens, range(1, len(tokens) + 1)))
        id2token = dict(zip(range(1, len(tokens) + 1), tokens))
        label2id = dict(zip(labels, range(len(labels))))
        id2label = dict(zip(range(len(labels)), labels))

        id2token[0] = self.PADDING
        token2id[self.PADDING] = 0

        id2token[len(tokens) + 1] = self.UNKNOWN
        token2id[self.UNKNOWN] = len(tokens) + 1

        with open(self.token2id_file, mode='w', encoding='utf-8') as outfile:
            for idx in id2token:
                outfile.write(id2token[idx] + '\t' + str(idx) + '\n')

        with open(self.label2id_file, mode='w', encoding='utf-8') as outfile:
            for idx in id2label:
                outfile.write(id2label[idx] + '\t' + str(idx) + '\n')

        return token2id, id2token, label2id, id2label

    def next_batch(self, x, y, start_index):
        last_index = start_index + self.batch_size
        x_batch = list(x[start_index:min(last_index, len(x))])
        y_batch = list(y[start_index:min(last_index, len(x))])
        if last_index > len(x):
            left_size = last_index - len(x)
            for i in range(left_size):
                index = np.random.randint(len(x))
                x_batch.append(x[index])
                y_batch.append(y[index])

        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        return x_batch, y_batch

    def padding(self, sample):
        for i in range(len(sample)):
            if len(sample[i]) < self.max_sequence_length:
                sample[i] += [self.token2id[self.PADDING] for _ in range(self.max_sequence_length - len(sample[i]))]
        return sample

    def prepare(self, df, is_padding=True):
        self.logger.info('loading data...')
        x = []
        y = []
        for index, record in tqdm(df.iterrows()):
            sentence = record.sentence.replace(' ', '')
            label = record.label[0]
            if self.token_level == 'word':
                sentence = self.word2vec_utils.processing_sentence(sentence, self.word2vec_utils.get_stop_word())
            else:
                sentence = list(record.sentence)
                if len(sentence) > self.max_sequence_length:
                    sentence = sentence[:self.max_sequence_length]
            # if len(sentence) != self.max_sequence_length:
            #     print(len(sentence))
            # assert len(sentence) == self.max_sequence_length
            tokens = []
            for word in sentence:
                if word in self.token2id:
                    tokens.append(self.token2id[word])
                else:
                    tokens.append(self.token2id[self.UNKNOWN])
            x.append(tokens)
            y.append(self.label2id[label])

        if is_padding:
            x = self.padding(x)

        return np.array(x), np.array(y)

    def get_training_set(self, ratio=0.9):
        df_train = read_csv(self.train_file, names=['sentence', 'label'], delimiter=',')
        x, y = self.prepare(df_train)
        num_samples = len(x)
        if self.dev_file is not None:
            x_train = x
            y_train = y
            x_val, y_val = self.get_valid_set()
        else:
            x_train = x[:int(num_samples * ratio)]
            y_train = y[:int(num_samples * ratio)]
            x_val = x[int(num_samples * ratio):]
            y_val = y[int(num_samples * ratio):]
            self.logger.info('validation set is not exist, built...')
        self.logger.info('training set size:{}, validation set size:{}'. format(len(x_train), len(x_val)))
        return x_train, y_train, x_val, y_val

    def get_training_dataset(self):
        x_train, y_train, x_val, y_val = self.get_training_set(ratio=0.8)
        # print(x_train)
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        valid_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        return train_dataset, valid_dataset

    def get_testing_dataset(self):
        X_val, y_val = self.get_valid_set()
        valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        return valid_dataset

    def get_valid_set(self):
        df_val = read_csv(self.dev_file, names=['sentence', 'label'], delimiter=',')
        x_val, y_val = self.prepare(df_val)
        return x_val, y_val

    def map_func(self, x, token2id):
        if str(x) == str(np.nan):
            return -1
        elif x not in token2id:
            return token2id[self.UNKNOWN]
        else:
            return token2id[x]

    def prepare_single_sentence(self, sentence):
        pass
