from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
import jieba
import multiprocessing
import pandas as pd
import os


class Word2VecUtils:
    def __init__(self, configs, logger):
        self.logger = logger
        self.stop_word_file = configs.stop_word_file
        self.train_data = configs.w2v_train_data
        self.model_dir = configs.w2v_model_dir
        self.model_name = configs.w2v_model_name
        self.model_path = os.path.join(self.model_dir, self.model_name)
        self.dim = configs.w2v_model_dim
        self.min_count = configs.w2v_min_count
        self.sg = 1 if configs.sg == 'skip-gram' else 0
        # softmax = 1
        self.hs = 1

    @staticmethod
    def processing_sentence(x, stop_words):
        cut_word = jieba.cut(str(x).strip())
        if stop_words:
            words = [word for word in cut_word if word not in stop_words and word != ' ']
        else:
            words = list(cut_word)
            words = [word for word in words if word != ' ']
        return words

    def get_stop_word(self):
        stop_word_list = []
        try:
            with open(self.stop_word_file, mode='r', encoding='utf-8') as stop_words_file:
                for line in stop_words_file:
                    stop_word_list.append(line.strip())
        except FileNotFoundError:
            return stop_word_list
        return stop_word_list

    def train_word2vec(self):
        df_train = pd.read_csv(self.train_data, encoding='utf-8')
        stop_words = self.get_stop_word()
        self.logger.info('Cut sentence...')
        tqdm.pandas(desc='Cut sentences...')
        df_train['sentence'] = df_train.sentence.process_apply(self.processing_sentence, args=(stop_words,))
        df_train.dropna(inplace=True)
        all_cut_sentence = df_train.sentence.to_list()
        self.logger.info('Training word2vec...')
        w2v_model = Word2Vec(vector_size=self.dim, workers=multiprocessing.cpu_count(), min_count=self.min_count, sg=self.sg)
        w2v_model.build_vocab(all_cut_sentence)
        w2v_model.train(all_cut_sentence, total_examples=w2v_model.corpus_total_words, epochs=100)
        w2v_model.save(self.model_path)
