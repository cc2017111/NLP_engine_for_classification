import codecs as cs
import os

def get_hierar_relations(hierar_taxonomy, label_map):
    print(hierar_taxonomy)
    hierar_relations = {}
    new_label_map = {}
    new_label_map_file = "/home/being/PycharmProjects/NLP_engine_for_classification/data/vocabs/HMCN_label_map"
    if not os.path.exists(new_label_map_file):
        level1_list = []
        level2_list = []
        level3_list = []
        for label, idx in label_map.items():
            level1_list.append(label[0])
            level2_list.append(label[:3])
            level3_list.append(label)
            # new_label_map[label] = new_label_map.get(label, idx)
        level1_label_set = set(level1_list)
        level2_label_set = set(level2_list)
        level3_label_set = set(level3_list)
        i = 0
        for label in level1_label_set:
            new_label_map[label] = str(i)
            i += 1
        for label in level2_label_set:
            new_label_map[label] = str(i)
            i += 1
        for label in level3_label_set:
            new_label_map[label] = str(i)
            i += 1
        with open(new_label_map_file, mode='w', encoding='utf-8') as outfile:
            for idx in new_label_map:
                outfile.write(str(new_label_map[idx]) + '\t' + str(idx) + '\n')
    else:
        with open(new_label_map_file, mode='r', encoding='utf-8') as file:
            rows = file.readlines()
            for row in rows:
                label = row.split('\t')[1].strip()
                id = row.split('\t')[0]
                new_label_map[label] = id

    print(new_label_map)
    hierarchical_class = [0, 0, 0]
    for label, idx in new_label_map.items():
        if len(label) == 1:
            hierarchical_class[0] += 1
        if len(label) == 3:
            hierarchical_class[1] += 1
        if len(label) == 4:
            hierarchical_class[2] += 1
    print(hierarchical_class)
    with cs.open(hierar_taxonomy, "r", encoding='utf-8') as f:
        for line in f:
            line_split = line.strip("\n").split(":")
            parent_label, children_label = line_split[0], line_split[1].split("\t")
            # print(parent_label, children_label)
            if parent_label not in new_label_map:
                continue
            parent_label_id = new_label_map[parent_label]
            children_label_ids = [new_label_map[child_label] for child_label in children_label if child_label in new_label_map]
            hierar_relations[parent_label_id] = children_label_ids
    return hierarchical_class, hierar_relations
