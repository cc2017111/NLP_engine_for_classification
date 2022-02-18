import os
import random

import jieba
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import BertTokenizer
from engines.utils.io_functions import read_csv
base_path = Path(__file__).resolve().parent.parent


class BertDataManager:
    def __init__(self, configs, logger):
        self.configs =  configs
        self.logger = logger

        self.train_file = str(base_path) + '/' + configs.datasets_fold + '/' + configs.train_file
        if configs.dev_file is not None:
            self.dev_file = configs.datasets_fold + '/' + configs.dev_file
        else:
            self.dev_file = None

        self.PADDING = '[PAD]'
        self.batch_size = configs.batch_size
        self.max_sequence_length = configs.max_sequence_length
        self.vocabs_dir = configs.vocabs_dir
        self.label2id_file = configs.vocabs_dir + '/label2id'
        self.label2id, self.id2label = self.load_labels()

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
        df_train = read_csv(self.train_file, names=['id', 'date', 'label', 'sentence', 'keyword'], delimiter='_!_')
        labels = list(set(df_train['label'][df_train['label'].notnull()]))
        id2label = dict(zip(range(0, len(labels)), labels))
        label2id = dict(zip(labels, range(0, len(labels))))
        with open(self.label2id_file, mode='w', encoding='utf-8') as outfile:
            for idx in id2label:
                outfile.write(str(id2label[idx]) + '\t' + str(idx) + '\n')

        return label2id, id2label

    def get_training_set(self, ratio=0.9):
        df_train = read_csv(self.train_file, names=['id', 'date', 'label', 'sentence', 'keyword'], delimiter='_!_')
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

    def get_valid_set(self):
        df_val =read_csv(self.dev_file, names=['id', 'date', 'label', 'sentence', 'keyword'], delimiter='_!_')
        X_val, y_val, att_mask_val, token_type_ids_val = self.prepare(df_val)
        return X_val, y_val, att_mask_val, token_type_ids_val

    def prepare(self, df):
        self.logger.info("loading data...")
        X= []
        y = []
        att_mask = []
        token_type_ids = []
        for index, record in tqdm(df.iterrows()):
            sentence = record.sentence
            label = record.label
            if len(sentence) < self.max_sequence_length - 2:
                tmp_x = self.tokenizer.encode(sentence)
                tmp_att_mask = [1] * len(tmp_x)
                tmp_y = self.label2id[label]

                tmp_x += [0 for _ in range(self.max_sequence_length - len(tmp_x))]
                tmp_att_mask += [0 for _ in range(self.max_sequence_length - len(tmp_att_mask))]
                token_type_ids = self.max_sequence_length * [0]
                X.append(tmp_x)
                y.append(tmp_y)
                att_mask.append(tmp_att_mask)
                token_type_ids.append(token_type_ids)
            else:
                tmp_x = self.tokenizer.encode(sentence)
                tmp_x = tmp_x[:self.max_sequence_length - 2]
                tmp_y = self.label2id[label]
                att_mask_tmp = [1] * self.max_sequence_length
                token_type_ids_tmp = [0] * self.max_sequence_length
                X.append(tmp_x)
                y.append(tmp_y)
                att_mask.append(att_mask_tmp)
                token_type_ids.append(token_type_ids_tmp)

        return np.array(X), np.array(y), np.array(att_mask), np.array(token_type_ids)

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


class DataManager:
    def __init__(self, configs, logger):
        self.configs = configs
        self.logger = logger

        self.UNKNOWN = '[UNK]'
        self.PADDING = '[PAD]'

        self.train_file = configs.datasets_fold + '/' + configs.train_file
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        if configs.dev_file is not None:
            self.dev_file = configs.datasets_fold + '/' + configs.dev_file
        else:
            self.dev_file = None

        self.batch_size = configs.batch_size
        self.max_sequence_length = configs.max_sequence_length
        self.embedding_dim = configs.embedding_dim
        self.vocabs_dir = configs.vocabs_dir
        self.token2id_file = configs.vocabs_dir + '/token2id'
        self.label2id_file = configs.vocabs_dir + '/label2id'

        self.token2id, self.id2token, self.label2id, self.id2label = self.load_vocab()
        self.max_sequence_length = configs.max_sequence_length
        self.max_token_num = len(self.tokenizer.get_vocab())
        self.max_label_num = len(self.label2id)
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
                token, token_id = row.split('¤')[0], int(row.split('¤')[1])
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
        df_train = read_csv(self.train_file, names=['id', 'date', 'label', 'sentence', 'keyword'], delimiter='_!_')
        tokens_dep = list(set(df_train['sentence'][df_train['sentence'].notnull()]))
        tokens = []
        for item in tokens_dep:
            for it in item:
                tokens += it
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
                outfile.write(id2token[idx] + '¤' + str(idx) + '\n')

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
            sentence = record.sentence
            label = record.label
            tmp_x = self.tokenizer.encode(sentence)
            tmp_y = self.label2id[label]
            if len(sentence) <= self.max_sequence_length:
                x.append(tmp_x)
            else:
                x.append(tmp_x[:self.max_sequence_length])
            y.append(tmp_y)

        if is_padding:
            x = np.array(self.padding(x))
        else:
            x = np.array(x)
        y = np.array(y)
        return x, y

    def get_training_set(self, ratio=0.9):
        df_train = read_csv(self.train_file, names=['id', 'date', 'label', 'sentence', 'keyword'], delimiter='_!_')
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
        self.logger.info('training set size:{}, validation set siez:{}'. format(len(x_train), len(x_val)))
        return x_train, y_train, x_val, y_val

    def get_valid_set(self):
        df_val = read_csv(self.train_file, names=['id', 'date', 'label', 'sentence', 'keyword'], delimiter='_!_')
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
