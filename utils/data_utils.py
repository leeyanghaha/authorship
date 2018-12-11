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
import utils.photo_utils as phu
import json


class SyntaxPathHolder:
    def __init__(self, word):
        self.path = []
        self.word = word


class UserHelper:
    def __init__(self, data_type):
        self.data_type = data_type

    def sample_user(self, users, sample_num):
        if len(users) < sample_num:
            return users
        else:
            return users[:sample_num]

    def count_user_number(self, data):
        counter = Counter()
        if self.data_type == ku.review:
            key = ku.reviewer_ID
        else:
            key = ku.twitter_id_str
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
    def __init__(self, data_type):
        self.data_type = data_type

    def get_text(self, data):
        text = []
        if self.data_type == ku.review:
            key = ku.review_text
        else:
            key = ku.twitter_text
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


class TorchDataLoader():
    def __init__(self, data_loader, data_arr, user2idx, ngram2idx, pos2idx, max_ngram_num,
                 max_pos_num, max_words_num ):
        self.data_loader = data_loader
        self.data_type = data_loader.data_type
        self.data_arr = data_arr
        self.user2idx = user2idx
        self.ngram2idx = ngram2idx
        self.pos2idx = pos2idx
        self.max_ngram_num = max_ngram_num
        self.max_pos_num = max_pos_num
        self.max_words_num = max_words_num
        self.nlp = spacy.load('en')
        self.datahelper = DataHelper(self.data_type)

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
        if self.data_type == ku.review:
            user = item[ku.reviewer_ID]
            text_key = ku.review_text
        else:
            user = item[ku.twitter_id_str]
            text_key = ku.twitter_text
        text_n_grams = self.datahelper.text2ngramid(item[text_key], self.ngram2idx)
        user_id = self.user2idx[user]
        ngram_id = self.datahelper.padding(text_n_grams, self.max_ngram_num)
        pos_id = self.get_pos_id(item[text_key], self.max_words_num, self.max_pos_num)
        position_id = self.get_position_id(pos_id)
        sample = {ku.pos_id: pos_id, ku.ngram_id: ngram_id, ku.user_id: user_id, ku.pos_order_id: position_id}
        return sample


class DataLoader:
    def __init__(self, data_type, review_domain=None, min_threshold=None, num_reviews_per_user=None):
        self.data_type = data_type
        self.review_domain = review_domain
        self.min_threshold = min_threshold
        self.num_reviews_per_user = num_reviews_per_user
        self.datahelper = DataHelper(self.data_type)

    def input_error_testing(self):
        if self.data_type == ku.review and self.review_domain is None:
            raise ValueError('if your data type is review then argument \'review_domain\''
                             ' must be not None, but got None.')

    def load_labeled_data(self, data_arr, u2i):
        y = []
        x = []
        if self.data_type == ku.review:
            Id = ku.reviewer_ID
            text = ku.review_text
        else:
            Id = ku.twitter_id_str
            text = ku.twitter_text
        for item in data_arr:
            user = u2i[item[Id]]
            x.append(item[text])
            y.append(user)
        return x, y

    def load_n_gram_feature_label(self, data_arr, ngram2idx, user2idx, max_ngram_len=None,
                                  binary=True):
        '''
        将 data_arr 转换成为n_gram表示
        :param data_arr:
        :param ngram2idx:
        :param user2idx:
        :param to_sparse:
        :return: x: [sample, clo], y: [sample, ]
        '''
        if binary == False and max_ngram_len == None:
            raise ValueError('if n-gram is not binary feature, then max_ngram_len must be not None, but got None')

        # text_list, y = self.load_labeled_data(data_arr, user2idx)
        if binary == True:
            x, y = self._load_binary_n_gram_feature_label(data_arr, user2idx, ngram2idx)
        else:
            x, y = self._load_ngram_idx_feature_label(data_arr, user2idx, ngram2idx, max_ngram_len)
        return x, y


    def _load_binary_n_gram_feature_label(self, data_arr, user2idx, ngram2idx):
        text_list, y = self.load_labeled_data(data_arr, user2idx)
        sample_num = len(text_list)
        clo = len(ngram2idx)
        x = np.zeros((sample_num, clo), dtype=bool)
        for idx, text in enumerate(text_list):
            text_ngram_id = self.datahelper.text2ngramid(text, ngram2idx)
            for i in range(len(text_ngram_id)):
                x[idx, int(text_ngram_id[i])] = True
        x = sparse.csr_matrix(x)
        return x, np.array(y)

    def _load_ngram_idx_feature_label(self, data_arr, user2idx, ngram2idx, max_ngram_len):
        text_list, y = self.load_labeled_data(data_arr, user2idx)
        sample_num = len(text_list)
        x = np.zeros((sample_num, max_ngram_len), dtype=np.uint32)
        for idx, text in enumerate(text_list):
            text_ngram_id = self.datahelper.text2ngramid(text, ngram2idx, padding=True, max_len=max_ngram_len)
            x[idx, :] = text_ngram_id
        return x, np.array(y)

    def syntax_get_feature_label(self, data_arr, user2idx, ngram2idx, pos2idx, max_ngram_num,
                 max_pos_num, max_words_num):
        # 因为torch中的DataSet类很好用，适合有复杂的输入，因此当输入比较复杂时将DataLoader转换为Torch中的DataSet
        torch_data_loader = TorchDataLoader(self, data_arr, user2idx, ngram2idx, pos2idx, max_ngram_num,
                                           max_pos_num, max_words_num)
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


class ReviewDataLoader(DataLoader):
    def __init__(self, data_type, review_domain, min_threshold, num_reviews_per_user):
        super(ReviewDataLoader, self).__init__(data_type, review_domain, min_threshold, num_reviews_per_user)
        self.input_error_testing()

    def load_domain_reviews(self):
        domain_dir = os.path.join(ku.user_root, self.review_domain)
        users = fu.listchildren(domain_dir, concat=False)
        all = []
        for user in users:
            all.extend(self._load_user_reviews(user))
        all = sku.shuffle(all)
        return all

    def _load_user_reviews(self, user):
        # 获取某个user, 某个domain下满足参数的reviews
        res = []
        file = os.path.join(ku.user_root, self.review_domain, user)
        reviews = fu.load_array(file)
        if self.min_threshold is not None:
            for review in reviews:
                if self.min_threshold <= review[ku.reviewer_count]:
                    res.append(review)
        else:
            res = reviews
        if self.num_reviews_per_user is not None:
            res = res[: self.num_reviews_per_user]
        return sku.shuffle(res)

    def load_users_data(self, users):
        res = []
        for user in users:
            res.extend(self._load_user_reviews(user))
        return sku.shuffle(res)

    def load_products_id(self, products, product2idx):
        res = []
        for product in products:
            res.append(product2idx[product])
        return np.array(res)


class TwitterDataLoader(DataLoader):
    def __init__(self, data_type, num_reviews_per_user, review_domain=None, min_threshold=None):
        super(TwitterDataLoader, self).__init__(data_type, review_domain, min_threshold, num_reviews_per_user)
        self.input_error_testing()

    def _load_user_tweets(self, user):
        file = os.path.join(ku.twitter_process, user)
        res = fu.load_array(file)
        if self.min_threshold is not None and len(res) < self.min_threshold:
            return []
        else:
            if self.num_reviews_per_user is not None:
                res = res[: self.num_reviews_per_user]
        return sku.shuffle(res)

    def load_users_data(self, users):
        res = []
        for user in users:
            res.extend(self._load_user_tweets(user))
        return sku.shuffle(res)

    def load_tweets_image(self, users):
        res = []
        for user in users:
            res.extend(self._load_tweets_image(user))
        return sku.shuffle(res)

    def _load_tweets_image(self, user):
        tweets = self.load_users_data([user])
        image_files = fu.listchildren(os.path.join(ku.photo_dir, user))
        images_name = fu.get_suffix_list(image_files)
        res = []
        for tweet in tweets:
            photo_id = tweet[ku.entities][ku.media][0][ku.media_id_str]
            if photo_id in images_name:
                image_file = os.path.join(ku.photo_dir, user, photo_id + '.jpg')
                image = phu.load_photo(image_file)
                if image is not None:
                    res.append({ku.twitter_text: tweet[ku.twitter_text], ku.image: image,
                                ku.twitter_user: user})
        return res

    def get_users(self, user_num):
        assert user_num <= 1198
        users = fu.listchildren(ku.twitter_process, concat=False)
        res = users[:user_num]
        # user_index = np.random.randint(0, len(users), user_num)
        # res = []
        # for i in user_index:
        #     res.append(users[i])
        return res



