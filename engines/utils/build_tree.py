import argparse
from pathlib import Path
import os
from configure import Configure

base_path = Path(__file__).resolve().parent.parent.parent
parser = argparse.ArgumentParser(description='BERT')
parser.add_argument('--config_file', default='system.config')
args = parser.parse_args()
configs = Configure(config_file=os.path.join(base_path, args.config_file))


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

label_level1, _ = load_labels_level_n(1)
label_level2, _ = load_labels_level_n(2)
label_level3, _ = load_labels_level_n(3)
label_level1 = label_level1.keys()
label_level2 = label_level2.keys()
label_level3 = label_level3.keys()
tree_label_file = label2id_file = str(base_path) + '/' + configs.vocabs_dir + '/tree_label_file'
with open(tree_label_file, mode='w', encoding='utf-8') as file:
    file.write('root' + ':' + '\t'.join(label_level1) + '\n')
    for level1_label in label_level1:
        temp_level2 = []
        for level2_label in label_level2:
            if level2_label.startswith(level1_label):
                temp_level2.append(level2_label)
        file.write(level1_label + ':' + '\t'.join(temp_level2) + '\n')

    for level2_label in label_level2:
        temp_level3 = []
        for level3_label in label_level3:
            if level3_label.startswith(level2_label):
                temp_level3.append(level3_label)
        file.write(level2_label + ':' + '\t'.join(temp_level3) + '\n')
