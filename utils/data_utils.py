import utils.key_utils as ku
import utils.function_utils as fu
from utils.multiprocess_utils import Multiprocess
import sklearn.utils as sku
from collections import  Counter
from scipy import sparse
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer


class Ngram(object):
    def __init__(self, n=3):
        self.n = n

    def character_level(self, text, n=3):
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


class SyntaxPathHolder:
    def __init__(self, word):
        self.path = []
        self.word = word


class UserHelper:

    def sample_user(self, users, sample_num):
        if len(users) < sample_num:
            return users
        else:
            return users[:sample_num]

    def count_number(self, data, key):
        counter = Counter()
        for item in data:
            user = item[key]
            counter.update([user])
        return counter

    def get_users(self, data, sample=None):
        user_counter = self.count_number(data, key=ku.reviewer_ID)
        users = np.array(list(user_counter.keys()))
        if sample:
            users = self.sample_user(users, sample)
        return users

    def user2idx(self, users):
        res = {}
        for idx, user in enumerate(set(users)):
            res.update({user: idx})
        return res

    def pos2idx(self):
        path = os.path.join(ku.voca_root, ku.pos2idx)
        return fu.load_array(path)[0]


class DataHelper:
    def get_text(self, data):
        text = []
        key = ku.review_text
        for item in data:
            text.append(item[key])
        return text

    def get_products(self, reviews):
        asin = []
        for review in reviews:
            asin.append(review[ku.asin])
        return asin

    def _get_max_character_n_gram_len(self, *reviews):
        ngram = Ngram()
        len_list = []
        reviews_text = self.get_text(reviews)
        for text in reviews_text:
            ngrams = ngram.character_level(text)
            len_list.append(len(ngrams))
        return len_list

    def get_max_len(self, reviews):
        mp = Multiprocess()
        res_getter = mp.multi_process(self._get_max_character_n_gram_len, arg_list=reviews)
        len_list = []
        for res in res_getter:
            len_list.extend(res)
        return max(len_list)

    def text2ngramid(self, text, ngram2idx, padding=False, max_len=None):
        ngram = Ngram()
        ngram_list = ngram.character_level(text)
        text_id = []
        for gram in ngram_list:
            if gram in ngram2idx:
                text_id.append(ngram2idx[gram])
            else:
                text_id.append(ngram2idx[ku.UNK])
        if padding:
            assert max_len is not None
            text_id = self.padding(text_id, max_len)
        return text_id

    def padding(self, text_id, max_ngram_len):
        if len(text_id) > max_ngram_len:
            text_id = text_id[:max_ngram_len]
        else:
            for i in range(len(text_id), max_ngram_len):
                text_id.append(ku.PAD)
        return text_id

    def product2idx(self, products):
        res = {}
        asin = set(products)
        for idx, item in enumerate(asin):
            res.update({item: idx})
        return res

    def load_products_id(self, products, product2idx):
        res = []
        for product in products:
            res.append(product2idx[product])
        return np.array(res)


class FeatureLoader:
    def __init__(self, **params):
        self.datahelper = DataHelper()
        self._check_params(**params)

    def _check_params(self, **params):
        if ku.feature2idx in params:
            self.feature2idx = params[ku.feature2idx]
        if ku.user2idx in params:
            self.user2idx = params[ku.user2idx]
        if 'pos2idx' in params:
            self.pos2idx = params[ku.pos2idx]
        if 'max_ngram_len' in params:
            self.max_ngram_len = params[ku.max_ngram_len]
        if 'max_pos_num' in params:
            self.max_pos_num = params[ku.max_pos_num]
        if 'max_words_num' in params:
            self.max_words_num = params[ku.max_words_num]

    def load_labeled_data(self, data_arr):
        y = []
        x = []
        Id = ku.reviewer_ID
        text = ku.review_text
        for item in data_arr:
            user = self.user2idx[item[Id]]
            x.append(item[text])
            y.append(user)
        return x, y

    def load_n_gram_binary_feature_label(self, data_arr, sparse_tag=True):
        text_list, y = self.load_labeled_data(data_arr)
        sample_num = len(text_list)
        clo = len(self.feature2idx)
        x = np.zeros((sample_num, clo + 1), dtype=bool)
        for idx, text in enumerate(text_list):
            text_ngram_id = self.datahelper.text2ngramid(text, self.feature2idx)
            for i in range(len(text_ngram_id)):
                x[idx, int(text_ngram_id[i])] = True
        if sparse_tag:
            x = sparse.csr_matrix(x)
        return x, np.array(y)

    def load_n_gram_idx_feature_label(self, data_arr, padding):
        text_list, y = self.load_labeled_data(data_arr)
        sample_num = len(text_list)
        x = np.zeros((sample_num, self.max_ngram_len), dtype=np.int32)
        for idx, text in enumerate(text_list):
            text_ngram_id = self.datahelper.text2ngramid(text, self.feature2idx, padding=padding, max_len=self.max_ngram_len)
            x[idx, :] = text_ngram_id
        return x, np.array(y)


def show_message(reviews):
    userhelper = UserHelper()
    datahelper = DataHelper()
    users = userhelper.get_users(reviews)
    products = datahelper.get_products(reviews)
    len_users = len(set(list(users)))
    products_num = len(set(list(products)))
    len_reviews = len(reviews)

    print('每条 product 有 {:.2f} 条 reviews '.format(len_reviews / products_num))
    print('users num', len_users)
    print('每个 user 有 {:.2f} reviews'.format(len_reviews / len_users))
    print('products num: ', products_num)
    print('共有 {} 条 reviews.'.format(len_reviews))


class ReviewLoader:
    def __init__(self, domain, product_num):
        self.domain = domain
        self.seed_product = 'B003EYVXV4'
        self.datahelper = DataHelper()
        self.userhelper = UserHelper()
        self.product2user = fu.load_array(os.path.join(ku.index_root, domain, 'product2user.json'))[0]
        self.user2product = fu.load_array(os.path.join(ku.index_root, domain, 'user2product.json'))[0]
        self.all_products = fu.listchildren(os.path.join(ku.product_root, domain), concat=False)
        self.product_num = product_num

    def _remove_rare_users(self, result, user_counter, threshold):
        removing_users = set()
        for user, count in dict(user_counter).items():
            if count < threshold:
                removing_users.add(user)
        for pro, reviews in result.items():
            temp = []
            for review in reviews:
                if review[ku.reviewer_ID] not in removing_users:
                    temp.append(review)
            result[pro] = temp
        return result

    def _check_efficiency(self, result):
        reviews = []
        for i in result:
            reviews.extend(result[i])
        show_message(reviews)
        return reviews

    def get_data(self):
        candidate_products = set(self.all_products)
        product = self.seed_product
        candidate_products.remove(product)
        product_counter = Counter()
        user_counter = Counter()
        result = {}
        next_products = []
        for i in range(self.product_num):
            next_product, product_counter_new, user_counter_new, result_new = self._iteration(product, product_counter,
                                                                                        user_counter,
                                                                                        self.user2product, result,
                                                                                        candidate_products)
            if next_product != '':
                candidate_products.remove(next_product)
                next_products.append(next_product)
                product = next_product
                product_counter = product_counter_new
                user_counter = user_counter_new
                result = result_new
            else:
                continue

        result = self._remove_rare_users(result, user_counter, threshold=20)
        reviews = self._check_efficiency(result)
        reviews = sku.shuffle(reviews)
        return reviews

    def get_product_reviews(self, product):
        path = os.path.join(ku.product_root, self.domain, product)
        reviews = fu.load_array(path)
        return reviews

    def _iteration(self, product, product_counter, user_counter, user2product,
                  result, candidate_products):
        reviews = self.get_product_reviews(product)
        users = self.userhelper.get_users(reviews)
        for user in users:
            product_counter.update(user2product[user])
        user_counter.update(users)
        result.update({product: reviews})
        most_product = product_counter.most_common()
        next_product = ''
        for p in most_product:
            p = p[0]
            if p in candidate_products:
                next_product = p
                break
            else:
                continue
        return next_product, product_counter, user_counter, result

    def load_products_id(self, products, product2idx):
        res = []
        for product in products:
            res.append(product2idx[product])
        return np.array(res)


def get_users_data(reviews, num_users):
    userhelper = UserHelper()
    user_counter = userhelper.count_number(reviews, key=ku.reviewer_ID)
    users = user_counter.most_common(num_users)
    users_set = set(item[0] for item in users)
    res = []
    for review in reviews:
        if review[ku.reviewer_ID] in users_set:
            res.append(review)
    show_message(res)
    return res



