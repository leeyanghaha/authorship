import utils.key_utils as ku
import utils.function_utils as fu
from utils.data_utils import FeatureLoader, UserHelper, DataHelper
from utils.vocabulary_utils import Vocabulary
import numpy as np
from scipy import sparse


class ReviewInfo:
    def __init__(self, reviews, feature='n-gram'):
        self.reviews = reviews
        self.feature = feature
        self.vocab = Vocabulary(ku.voca_root)
        self.feature2idx = self.feature2idx(6)
        self.product2idx = self.product2idx()
        self.user2idx = self.user2idx()
        self.user_num = len(self.user2idx)
        self.product_num = len(self.product2idx)
        self.x, self.users = self.feature_label()
        self.products = self.get_products()

    def user2idx(self):
        userhelper = UserHelper()
        users = userhelper.get_users(self.reviews)
        return userhelper.user2idx(users)

    def product2idx(self):
        datahelper = DataHelper()
        products = datahelper.get_products(self.reviews)
        return datahelper.product2idx(products)

    def feature2idx(self, min_threshold):
        '''get a dict that convert each word to feature'''
        assert self.feature == 'n-gram' or self.feature == 'word'
        if self.feature == 'n-gram':
            feature2idx = self.vocab.character_n_gram_table(self.reviews,
                                                            min_threshold=min_threshold)
        else:
            feature2idx = self.vocab.word_table(self.reviews, min_threshold=0)
        return feature2idx

    def feature_label(self):
        '''convert text to feature and get it's label'''
        data_params = {'user2idx': self.user2idx, 'ngram2idx': self.feature2idx}
        feature_loader = FeatureLoader(**data_params)
        x, y = feature_loader.load_n_gram_binary_feature_label(self.reviews)
        return x, y

    def get_products(self):
        products = []
        for review in self.reviews:
            products.append(self.product2idx[review[ku.asin]])
        return products

    def split_data(self):
        training_split = int(len(self.reviews) * 0.6)
        valid_split = training_split + int(len(self.reviews) * 0.2)
        training_x, training_y = self.x[: training_split], self.users[: training_split]
        valid_x, valid_y = self.x[training_split: valid_split], self.users[training_split: valid_split]
        test_x, test_y = self.x[valid_split:], self.users[valid_split:]
        return (training_x, training_y), (valid_x, valid_y), (test_x, test_y)
