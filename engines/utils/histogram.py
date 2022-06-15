import argparse
import collections
from pathlib import Path
import os
from collections import Counter
from configure import Configure
from utils.logger import get_logger
from data import BertDataManager, DataManager
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from engines.train import train
base_path = Path(__file__).resolve().parent.parent.parent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT')
    parser.add_argument('--config_file', default='system.config')
    args = parser.parse_args()
    configs = Configure(config_file=os.path.join(base_path, args.config_file))
    logger = get_logger(configs.log_dir)
    mode = configs.mode.lower()
    if configs.model == "BERT":
        dataManager = BertDataManager(configs, logger)
    else:
        dataManager = DataManager(configs, logger)

    y_train = dataManager.get_labels()
    res = dict(collections.Counter(y_train))
    print(res)
    count = 0
    k_list = []
    print(dataManager.id2label)
    for k, v in res.items():
        if v <= 100:
            k_list.append(dataManager.id2label[str(k)])
            count += 1
    print(k_list)
    print(count)
    labels = set(y_train)
    groups = len(labels)
    labels_ori = list(dataManager.id2label.values())
    plt.figure(figsize=(20, 8), dpi=80)
    n, bins, patches = plt.hist(y_train, bins=groups, color='b', edgecolor='k', rwidth=0.8, range=(0, groups), align='left', label="histogram")
    # for i in range(len(n)):
    #     plt.text(bins[i], n[i] * 1.02, int(n[i]), fontsize=8, horizontalalignment="center")
    # plt.xticks(list(range(groups)), labels_ori, fontsize=8, rotation=-90)
    plt.xticks([])
    plt.yticks(fontsize=15)
    plt.xlabel('class level 3', fontsize=20)
    plt.ylabel('num', fontsize=20)
    # plt.grid(True, linestyle='--', alpha=1)
    plt.title('histogram of IPC class level 3')
    plt.tight_layout()
    plt.show()

    # sns.displot(y_train)

