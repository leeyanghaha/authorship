import time
import math
import multiprocessing as mp
import utils.pattern_utils as pu
import utils.key_utils as ku
import re
import json
import os


def text_normalization(text):
    text = pu.text_normalization(text)
    text = ' '.join(tokenize(text))
    return text


def tokenize(text):
    word_list = []
    tokenize_pattern = r"[a-zA-Z0-9]+(?:['_-][a-zA-Z0-9]+)*"
    for word in text.lower().strip().split():
        match = re.search(tokenize_pattern, word)
        if match:
            word_list.append(match.group())
    return word_list



def load_array(file):
    with open(file, encoding='utf-8', errors='ignore') as f:
        array = [json.loads(line) for line in f.readlines()]
    return array


def load_file(file):
    res = []
    with open(file) as f:
        for line in f:
            res.append(line.strip())
    return res


def dump_file(array, file):
    if os.path.exists(file):
        print('rm {}.'.format(file))
        os.system('rm {}'.format(file))
    with open(file, 'a') as f:
        for i in array:
            f.write(json.dumps(i) + '\n')


def listchildren(dir, concat=True):
    if not os.path.exists(dir):
        raise ValueError('no such dir')
    children_list = sorted(os.listdir(dir))
    if not concat:
        return children_list
    else:
        return [os.path.join(dir, file) for file in children_list]


def dump_array(tw_array, file):
    with open(file, 'a') as f:
        for tw in tw_array:
             f.write(json.dumps(tw) + '\n')


def get_suffix(file):
    return os.path.split(file)[1]


def get_suffix_list(files):
    res = []
    for file in files:
        res.append(os.path.splitext(get_suffix(file))[0])
    return res


def get_stopwords():
    path = os.path.join(ku.root, 'code/utils/stopwords.txt')
    return load_file(path)


if __name__ == '__main__':
    pass
