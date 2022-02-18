import json
import csv
import pandas as pd


def read_csv(file_name, names, delimiter='t'):
    with open(file_name, 'r', encoding='utf-8') as file_obj:
        for line in file_obj.readlines():
            line.replace('\n', '').replace(' ', '\t')

    if delimiter == 't':
        sep = '\t'
    elif delimiter == 'b':
        sep = ' '
    else:
        sep = delimiter
    return pd.read_csv(file_name, sep=sep, quoting=csv.QUOTE_NONE, encoding='utf-8', skip_blank_lines=False, header=None, names=names)