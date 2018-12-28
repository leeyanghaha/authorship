import utils.key_utils as ku
import utils.function_utils as fu
from utils.multiprocess_utils import Multiprocess
import sklearn.utils as sku
from models.n_grams import Ngram
from collections import  Counter
from scipy import sparse
import numpy as np
import spacy
import os
from keras.utils.np_utils import to_categorical
import utils.photo_utils as phu
import json


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

    def count_user_number(self, data):
        counter = Counter()
        key = ku.reviewer_ID
        for item in data:
            user = item[key]
            counter.update([user])
        return counter

    def get_users(self, data, sample=None):
        user_counter = self.count_user_number(data)
        users = np.array(list(user_counter.keys()))
        if sample:
            users = self.sample_user(users, sample)
        return users

    def user2idx(self, users):
        res = {}
        for idx, user in enumerate(set(users)):
            res.update({user:idx})
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



def syntax(root, syntax_path, root_holder):
    if len(list(root.children)) == 0:
        return
    else:
        for child in root.children:
            child_holder = SyntaxPathHolder(child)
            child_holder.path.extend(root_holder.path + [child.pos_])
            syntax_path.append(child_holder)
            if len(list(child.children)) > 0:
                root_holder = child_holder
            syntax(child, syntax_path, root_holder)
        return syntax_path


def syntax_tree(nlp, text, pos2idx):
    doc = nlp(text)
    pos_id = []
    for sentence in list(doc.sents):
        syntax_path = []
        root = sentence.root
        root_holder = SyntaxPathHolder(root)
        root_holder.path.append(root.pos_)
        syntax_path = syntax(root, syntax_path, root_holder)
        if syntax_path is not None:
            for word_path in syntax_path:
                temp = []
                for pos in word_path.path:
                    if pos in pos2idx:
                        temp.append(pos2idx[pos])
                    else:
                        temp.append(pos2idx[ku.UNK])
                pos_id.append(temp)
    return pos_id


class ItemLoader():
    def __init__(self, data_arr, user2idx, ngram2idx, pos2idx, max_ngram_num,
                 max_pos_num, max_words_num ):
        self.data_arr = data_arr
        self.user2idx = user2idx
        self.ngram2idx = ngram2idx
        self.pos2idx = pos2idx
        self.max_ngram_num = max_ngram_num
        self.max_pos_num = max_pos_num
        self.max_words_num = max_words_num
        self.nlp = spacy.load('en')
        self.datahelper = DataHelper()

    def get_pos_id(self, text, max_words_num, max_pos_num):
        '''
        return 一个list的list, 其中每个list 代表一条 syntax path , len: len(tokenize(text))
        :param text:
        :return:
        '''
        pos_id = np.zeros((max_words_num, max_pos_num), dtype=np.uint32)
        text_pos_list = syntax_tree(self.nlp, text, self.pos2idx)
        words_len = len(text_pos_list)
        for i in range(pos_id.shape[0]):
            if i < words_len:
                word_path = self.datahelper.padding(text_pos_list[i], max_pos_num)
            else:
                word_path = [ku.PAD for _ in range(max_pos_num)]
            pos_id[i, :] = word_path
        return pos_id

    def get_position_id(self, pos_id):
        position_id = np.zeros((pos_id.shape[0], pos_id.shape[1]), dtype=np.uint32)
        for i in range(pos_id.shape[0]):
            position_i = [j+1 for j in range(len(pos_id[i, :])) if pos_id[i, j] != ku.PAD]
            position_i = self.datahelper.padding(position_i, position_id.shape[1])
            position_id[i, :] = position_i
        return position_id

    def __len__(self):
        return len(self.data_arr)

    def __getitem__(self, idx):
        item = self.data_arr[idx]
        user = item[ku.reviewer_ID]
        text_key = ku.review_text
        text_n_grams = self.datahelper.text2ngramid(item[text_key], self.ngram2idx)
        user_id = self.user2idx[user]
        ngram_id = self.datahelper.padding(text_n_grams, self.max_ngram_num)
        pos_id = self.get_pos_id(item[text_key], self.max_words_num, self.max_pos_num)
        position_id = self.get_position_id(pos_id)
        sample = {ku.pos_id: pos_id, ku.ngram_id: ngram_id, ku.user_id: user_id, ku.pos_order_id: position_id}
        return sample



class FeatureLoader:
    def __init__(self, **params):
        self.datahelper = DataHelper()
        self._check_params(**params)

    def _check_params(self, **params):
        if ku.ngram2idx in params:
            self.ngram2idx = params[ku.ngram2idx]
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
        clo = len(self.ngram2idx)
        x = np.zeros((sample_num, clo + 1), dtype=bool)
        for idx, text in enumerate(text_list):
            text_ngram_id = self.datahelper.text2ngramid(text, self.ngram2idx)
            for i in range(len(text_ngram_id)):
                x[idx, int(text_ngram_id[i])] = True
        if sparse_tag:
            x = sparse.csr_matrix(x)
        return x, np.array(y)

    def load_n_gram_idx_feature_label(self, data_arr):
        text_list, y = self.load_labeled_data(data_arr)
        sample_num = len(text_list)
        x = np.zeros((sample_num, self.max_ngram_len), dtype=np.uint32)
        for idx, text in enumerate(text_list):
            text_ngram_id = self.datahelper.text2ngramid(text, self.ngram2idx, padding=True, max_len=self.max_ngram_len)
            x[idx, :] = text_ngram_id
        return x, np.array(y)

    def syntax_cnn_feature_label(self, data_arr):
        torch_data_loader = ItemLoader(data_arr, self.user2idx, self.ngram2idx, self.pos2idx,
                                       self.max_ngram_len, self.max_pos_num, self.max_words_num)
        ngram_id = []
        pos_id = []
        position_id = []
        user_id = []
        reviews_num = len(torch_data_loader)
        for item in torch_data_loader:
            ngram_id.append(item[ku.ngram_id])
            pos_id.append(item[ku.pos_id])
            position_id.append(item[ku.pos_order_id])
            user_id.append(item[ku.user_id])
        return {ku.ngram_id: np.array(ngram_id),
                ku.pos_id: np.array(pos_id).flatten().reshape((reviews_num, -1)),
                ku.pos_order_id: np.array(position_id).flatten().reshape((reviews_num, -1)),
                ku.user_id: np.array(user_id)}

    def one_hot_encoding(self, ids, num_classes):
        res = []
        for id in ids:
            one_hot_id = to_categorical(id, num_classes=num_classes)
            res.append(one_hot_id)
        return np.array(res)


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
        users = self.userhelper.get_users(reviews)
        len_users = len(set(list(users)))
        len_reviews = len(reviews)

        print('每条 product 有 {:.2f} 条 reviews '.format(len_reviews / self.product_num))
        print('users num', len_users)
        print('每个 user 有 {:.2f} reviews'.format(len_reviews / len_users))
        print('products num: ', self.product_num)
        print('共有 {} 条 reviews.'.format(len_reviews))
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



