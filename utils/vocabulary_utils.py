import utils.data_utils as du
import utils.key_utils as ku
import utils.multiprocess_utils as mu
# from models.n_grams import Ngram
from collections import Counter
import os
import json
from sklearn.feature_extraction.text import CountVectorizer


class Ngram(object):
    def __init__(self, n=3):
        self.n = n

    def character_level(self, text, n=None):
        PSD = ' '
        PED = ' '
        text = PSD + text + PED
        n = self.n if n is None else n
        character_n_grams = [text[i:i+n] for i in range(len(text) - n + 1)]
        return character_n_grams

    def word_level(self, text, n=2):
        ngram_vector = CountVectorizer(ngram_range=(n, n), decode_error='ignore',
                                       token_pattern=r'\b\w+\b', min_df=1)
        ngram_vector.fit_transform([text])
        return list(ngram_vector.vocabulary_.keys())


class Vocabulary(object):
    def __init__(self, voca_root):
        self.ngram = Ngram()
        self.data_helper = du.DataHelper()
        self. voca_root = voca_root

    def counter_character_n_grams(self, *reviews):
        text_array = self.data_helper.get_text(reviews)
        # text_array = reviews
        counter = Counter()
        for text in text_array:
            ngrams = self.ngram.character_level(text)
            counter.update(ngrams)
        return counter

    def count_word(self, *reviews):
        text_array = self.data_helper.get_text(reviews)
        counter = Counter()
        for text in text_array:
            words = text.lower().strip().split(' ')
            counter.update(words)
        return counter

    def counter_word_n_gram(self, *reviews):
        # text_array = self.data_helper.get_text(reviews)
        text_array = reviews
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

    def multi_counter_word(self, reviews):
        mp = mu.Multiprocess()
        res_getter = mp.multi_process(self.count_word, arg_list=reviews)
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

    def remove_rare(self, counter, min_threshold):
        return {n_gram: count for n_gram, count in dict(counter).items() if count >= min_threshold}

    def character_n_gram_table(self, reviews, min_threshold):
        counter = self.multi_counter_character_n_grams(reviews)
        # counter = self.counter_character_n_grams(reviews)
        n_grams_count = self.remove_rare(counter, min_threshold)
        n_gram2idx = dict()
        idx2n_gram = dict()
        for idx, n_gram in enumerate(n_grams_count):
            n_gram2idx.update({n_gram:idx+1})
            idx2n_gram.update({idx+1:n_gram})
        n_gram2idx[ku.UNK] = 0
        idx2n_gram[0] = ku.UNK
        return n_gram2idx

    def word_table(self, reviews, min_threshold):
        counter = self.multi_counter_word(reviews)
        word_counter = self.remove_rare(counter, min_threshold)
        word2idx = dict()
        for idx, word in enumerate(word_counter):
            word2idx.update({word: idx+1})
        word2idx[ku.UNK] = 0
        return word2idx

    def word_n_gram_table(self, reviews, min_threshold, start):
        counter = self.multi_counter_word_n_grams(reviews)
        # counter = self.counter_word_n_gram(reviews)
        n_grams_count = self.remove_rare(counter, min_threshold)
        n_gram2idx = dict()
        for idx, n_gram in enumerate(n_grams_count):
            n_gram2idx.update({n_gram:idx+1+start})
        n_gram2idx[ku.UNK] = 0
        return n_gram2idx

    def dump_n_grams(self, n_grams_table, type):
        dump_path = os.path.join(self.voca_root, type)
        if os.path.exists(dump_path):
            print('rm {}'.format(dump_path))
            os.system('rm {}'.format(dump_path))
        else:
            print('ngrams dumped in {}.'.format(dump_path))
            with open(dump_path, 'a') as f:
                f.write(json.dumps(n_grams_table))

    def load_n_grams(self, type):
        load_path = os.path.join(self.voca_root, type)
        if os.path.exists(load_path):
            with open(load_path) as f:
                return json.loads(f.readline())
        else:
            raise ValueError('{} does not exit'.format(load_path))




