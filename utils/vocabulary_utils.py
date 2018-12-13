import utils.data_utils as du
import utils.key_utils as ku
import utils.multiprocess_utils as mu
from models.n_grams import Ngram
from collections import Counter
import os
import json


class Vocabulary(object):
    def __init__(self, voca_root):
        self.ngram = Ngram()
        self.data_helper = du.DataHelper()
        self. voca_root = voca_root

    def counter_character_n_grams(self, *reviews):
        text_array = self.data_helper.get_text(reviews)
        counter = Counter()
        for text in text_array:
            ngrams = self.ngram.character_level(text)
            counter.update(ngrams)
        return counter

    def counter_word_n_gram(self, *reviews):
        text_array = self.data_helper.get_text(reviews)
        counter = Counter()
        for text in text_array:
            if len(text) > 0:
                ngrams = self.ngram.word_level(text, n=2)
                counter.update(ngrams)
        return counter

    def multi_counter_character_n_grams(self, reviews):
        mp = mu.Multiprocess()
        res_getter = mp.multi_process(self.counter_character_n_grams, arg_list=reviews)
        counter = Counter()
        for every_res in res_getter:
            counter.update(every_res)
        return counter

    def multi_counter_word_n_grams(self, reviews):
        mp = mu.Multiprocess()
        res_getter = mp.multi_process(self.counter_word_n_gram, arg_list=reviews)
        counter = Counter()
        for every_res in res_getter:
            counter.update(every_res)
        return counter

    def remove_rare_n_grams(self, counter, min_threshold):
        return {n_gram:count for n_gram, count in dict(counter).items() if count >= min_threshold}

    def character_n_gram_table(self, reviews, min_threshold):
        counter = self.multi_counter_character_n_grams(reviews)
        n_grams_count = self.remove_rare_n_grams(counter, min_threshold)
        n_gram2idx = dict()
        idx2n_gram = dict()
        for idx, n_gram in enumerate(n_grams_count):
            n_gram2idx.update({n_gram:idx+1})
            idx2n_gram.update({idx+1:n_gram})
        n_gram2idx[ku.UNK] = 0
        idx2n_gram[0] = ku.UNK
        return n_gram2idx

    def word_n_gram_table(self, reviews, min_threshold, start):
        counter = self.multi_counter_word_n_grams(reviews)
        n_grams_count = self.remove_rare_n_grams(counter, min_threshold)
        n_gram2idx = dict()
        for idx, n_gram in enumerate(n_grams_count):
            n_gram2idx.update({n_gram:idx+1+start})
        n_gram2idx[ku.UNK] = 0
        return n_gram2idx


    def dump_n_grams(self, n_grams_table, type):
        assert type == ku.charngram2idx or type == ku.wordngram2idx
        dump_path = os.path.join(self.voca_root, type)
        if os.path.exists(dump_path):
            print('rm {}'.format(dump_path))
            os.system('rm {}'.format(dump_path))
        else:
            print('ngrams dumped in {}.'.format(dump_path))
            with open(dump_path, 'a') as f:
                f.write(json.dumps(n_grams_table))

    def load_n_grams(self, type):
        assert type == ku.charngram2idx or type == ku.wordngram2idx
        load_path = os.path.join(self.voca_root, type)
        if os.path.exists(load_path):
            with open(load_path) as f:
                return json.loads(f.readline())
        else:
            raise ValueError('{} does not exit'.format(load_path))




