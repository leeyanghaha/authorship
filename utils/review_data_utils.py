import utils.function_utils as fu
from utils.multiprocess_utils import Multiprocess
import utils.key_utils as ku
from collections import Counter
import os
import numpy as np
import json
from models.n_grams import Ngram
import  sklearn.utils as sku


def user_number(*data):
    counter = Counter()
    for review in data:
        if ku.reviewer_ID in review:
            user = review[ku.reviewer_ID]
            counter.update([user])
    return counter


def multi_user_counter(data):
    mp = Multiprocess(10)
    res_list = mp.multi_process(user_number, arg_list=data)
    result = {}
    for res in res_list:
        for user, count in dict(res).items():
            if user in result:
                result[user] += count
            else:
                result[user] = count
    return result


def user2idx(data):
    user_count = multi_user_counter(data)
    u2i = {}
    for idx, user in enumerate(user_count):
        u2i.update({user:idx})
    return u2i


def remove_user_less_than_K(K, reviews):
    res = []
    user_counter = multi_user_counter(reviews)
    candidate_users = {user for user, count in user_counter.items() if count >= K}
    print('user reviews larger than {}: {} users'.format(K, len(candidate_users)))
    for review in reviews:
        if review[ku.reviewer_ID] in candidate_users:
            review[ku.reviewer_count] = user_counter[review[ku.reviewer_ID]]
            res.append(review)
    return res


def sample_reviews(N, reviews, order=False):
    if order:
        return reviews[:N]
    else:
        random_index = np.random.randn(N)
        sample_reviews = []
        for i in random_index:
            sample_reviews.append(reviews[int(i)])
        return sample_reviews


def organise_by_user(source_root, domain):
    source_path = os.path.join(source_root, '{}.json'.format(domain))
    source_reviews = fu.load_array(source_path)
    user_dict = {}
    for review in source_reviews:
        user_ID = review[ku.reviewer_ID]
        if user_ID in user_dict:
            user_dict[user_ID].append(review)
        else:
            user_dict[user_ID] = [review]
    return user_dict


def dump_user_reviews(user_reviews, target_root, domain):
    target_domain_dir = os.path.join(target_root, domain)
    for user, reviews in user_reviews.items():
        target_path = os.path.join(target_domain_dir, user)
        fu.dump_file(reviews, target_path)



class DataLoader:
    def __init__(self, root, domain, min_threshold=None, max_threshold=None,
                 num_reviews_per_user=None, shuffle=True):
        self.root = root
        self.domain = domain
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.num_reviews_per_user = num_reviews_per_user
        self.shuffle = shuffle

    def load_domain_reviews(self):
        # 获取某一domain下所有user的reviews
        '''
        :param shuffle:
        :param min_threshold: 用户发表的最小评论数
        :param max_threshold: 用户发表的最大评论数
        :return:
        '''
        domain_dir = os.path.join(self.root, self.domain)
        users = fu.listchildren(domain_dir, concat=False)
        all = []
        for user in users:
            all.extend(self._load_user_reviews(user))
        if self.shuffle:
            all = sku.shuffle(all)
        return all

    def _load_user_reviews(self, user):
        # 获取某个user, 某个domain 下的reviews
        '''

        :param file:
        :param min_threshold: 用户发表的最小评论数
        :param max_threshold: 用户发表的最大评论数
        :return:
        '''
        res = []
        file = os.path.join(self.root, self.domain, user)
        with open(file) as f:
            reviews = f.readlines()
        if self.min_threshold is not None and self.max_threshold is not None:
            for review in reviews:
                review = json.loads(review)
                if self.min_threshold <= review[ku.reviewer_count] <= self.max_threshold:
                    res.append(review)
        elif self.min_threshold is not None:
            for review in reviews:
                review = json.loads(review)
                if review[ku.reviewer_count] >= self.min_threshold:
                    res.append(review)
        elif self.max_threshold is not None:
            for review in reviews:
                review = json.loads(review)
                if review[ku.reviewer_count] <= self.max_threshold:
                    res.append(review)
        else:
            res = fu.load_array(file)
        if self.num_reviews_per_user is not None and len(res) > self.num_reviews_per_user:
            res = res[:self.num_reviews_per_user]
        if self.shuffle:
            res = sku.shuffle(res)
        return res

    def load_users_reviews(self, users):
        res = []
        for user in users:
            res.extend(self._load_user_reviews(user))
        if self.shuffle:
            res = sku.shuffle(res)
        return res


    def load_labeled_data(self, reviews, u2i):
        y = []
        x = []
        for review in reviews:
            user = u2i[review[ku.reviewer_ID]]
            x.append(review[ku.review_text])
            y.append(user)
        return x, y


class DataHelper:
    def get_text(self, reviews):
        text = []
        for review in reviews:
            text.append(review[ku.review_text])
        return text

    def count_user_number(self, reviews):
        counter = Counter()
        for review in reviews:
            user = review[ku.reviewer_ID]
            counter.update([user])
        return counter

    def get_users(self, reviews, sample=None):
        user_counter = self.count_user_number(reviews)
        users = np.array(list(user_counter.keys()))
        if sample:
            users = UserHelper().sample_user(users, sample)
        return users


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

    def user2idx(self, users):
        res = {}
        for idx, user in enumerate(users):
            res.update({user:idx})
        return res


class UserHelper:
    def sample_user(self, users, sample_num):
        if len(users) < sample_num:
            return users
        else:
            return users[:sample_num]



if __name__ == '__main__':
    root = '/home/nfs/yangl/research/authorship/data/user'
    dataloader = DataLoader(root, ku.Kindle)
    data = dataloader.load_domain_reviews(min_threshold=500,
                                                           max_threshold=None)
    x, y = dataloader.load_labeled_data(data)
    print(x[0], y[0])
    # AWAP0KEX6POQV
    # ngram = Ngram()
    # counter = Counter()
    # for review in data:
    #     text = review[ku.review_text].strip()
    #     char_n_gram = ngram.character_level(text)
    #     counter.update(char_n_gram)
    # filter2 = {i for i, j in dict(counter).items() if j > 15}
    # filter3 = {i for i, j in dict(counter).items() if j > 3}
    # filter4 = {i for i, j in dict(counter).items() if j > 4}
    # filter5 = {i for i, j in dict(counter).items() if j > 5}
    # print(len(filter2), len(filter3), len(filter4), len(filter5))