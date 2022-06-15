# import jieba
# token2id, id2token = {}, {}
# with open("//data/vocabs/token2id", mode='r', encoding='utf-8') as infile:
#     for row in infile:
#         row = row.strip()
#         token, token_id = row.split('¤')[0], int(row.split('¤')[1])
#         token2id[token] = token_id
#         id2token[token_id] = token
# dep = "林徽因什么理由 拒绝了 徐志摩而选择梁思成为终身伴侣？"
# print(token2id[" "])

import numpy as np
# pe = np.array([[pos / np.power(10000, 2 * (i // 2) / 1024) for i in range(1024)] for pos in range(256)])
# print("o:", pe)
# pe[:, 0::2] = np.sin(pe[:, 0::2])
# pe[:, 1::2] = np.cos(pe[:, 1::2])
# print("l:", pe.shape)

# import argparse
# from pathlib import Path
# import os
# from engines.configure import Configure
# from engines.utils.logger import get_logger
# from engines.data import BertDataManager, DataManager
# from engines.utils.get_hierar_relations import get_hierar_relations
# base_path = Path(__file__).resolve().parent.parent.parent
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='BERT')
#     parser.add_argument('--config_file', default='system.config')
#     args = parser.parse_args()
#     configs = Configure(config_file=os.path.join(base_path, args.config_file))
#     logger = get_logger(configs.log_dir)
#     mode = configs.mode.lower()
#     if configs.model == "BERT":
#         dataManager = BertDataManager(configs, logger)
#     else:
#         dataManager = DataManager(configs, logger)
#     tree_label_file = str(base_path) + '/data/vocabs/tree_label_file'
#     print(dataManager.label2id)
#     print(get_hierar_relations(tree_label_file, dataManager.label2id))

import tensorflow as tf
a = tf.constant([410])
b = tf.squeeze(a)
print(b)
