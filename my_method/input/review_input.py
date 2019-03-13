import torch
from torch.utils.data import Dataset, DataLoader
import torch
import utils.key_utils as ku
import utils.function_utils as fu
from utils.data_utils import FeatureLoader, UserHelper, DataHelper
from utils.vocabulary_utils import Vocabulary
import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer
import json


def get_reviews():
    file = '/home/leeyang/research/data/Movie.json'
    return fu.load_array(file)


class ReviewInfo:
    def __init__(self, reviews):
        self.reviews = reviews
        self.product2idx = self.product2idx()
        self.user2idx = self.user2idx()
        self.user_num = len(self.user2idx)
        self.product_num = len(self.product2idx)

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
        raise NotImplementedError('Not Implemented')

    def feature_label(self):
        '''convert text to feature and get it's label'''
        raise NotImplementedError('Not Implemented')


class PreTrainedInfo(ReviewInfo):
    def __init__(self, reviews,
                 feature_file,
                 bert_vocab,
                 max_seq_len,
                 feature_dim):
        super(PreTrainedInfo, self).__init__(reviews)
        self.feature_file = feature_file
        self.bert_vocab = bert_vocab
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim
        self.x, self.users, self.products = self.feature_label()

    def tokenize(self):
        texts = []
        users = []
        products = []
        tokenizer = BertTokenizer.from_pretrained(self.bert_vocab, do_lower_case=True)
        for review in self.reviews:
            text = tokenizer.tokenize(review[ku.review_text])
            text = self._trunc_or_pad(text)
            texts.append(text)
            users.append(self.user2idx[review[ku.reviewer_ID]])
            products.append(self.product2idx[review[ku.asin]])
        return texts, users, products

    def _trunc_or_pad(self, text):
        if len(text) < self.max_seq_len:
            pad = ['PAD'] * (self.max_seq_len - len(text))
            text += pad
        else:
            text = text[: self.max_seq_len]
        return text

    def feature2idx(self, min_threshold):
        with open(self.feature_file) as f:
            line = f.readline()
            feature = json.loads(line)
        return feature

    def feature_label(self):
        feature = self.feature2idx(None)
        texts, users, products = self.tokenize()
        x = []
        for text in texts:
            x_text = []
            for word in text:
                if word in feature:
                    x_text.append(feature[word])
                else:
                    x_text.append([0.0] * self.feature_dim)
            x.append(x_text)
        return np.array(x), np.array(users), np.array(products)


class NonPreTrainedInfo(ReviewInfo):
    def __init__(self, reviews,
                 min_threshold=6,
                 feature_name='n-gram',
                 max_seq_len=3500
                 ):
        super(NonPreTrainedInfo, self).__init__(reviews)
        self.min_threshold = min_threshold
        self.feature = feature_name
        self.max_seq_len = max_seq_len
        self.vocab = Vocabulary(ku.voca_root)
        self.x, self.users = self.feature_label()
        self.products = self.get_products()

    def feature2idx(self, min_threshold):
        assert self.feature == 'n-gram' or self.feature == 'word'
        if self.feature == 'n-gram':
            feature2idx = self.vocab.character_n_gram_table(self.reviews,
                                                            min_threshold=min_threshold)
        else:
            feature2idx = self.vocab.word_table(self.reviews, min_threshold=0)
        return feature2idx

    def feature_label(self):
        feature2idx = self.feature2idx(self.min_threshold)
        data_params = {'max_ngram_len': self.max_seq_len, 'user2idx': self.user2idx, 'ngram2idx': feature2idx}
        feature_loader = FeatureLoader(**data_params)
        x, y = feature_loader.load_n_gram_idx_feature_label(self.reviews)
        return x, y

    def get_products(self):
        products = []
        for review in self.reviews:
            products.append(self.product2idx[review[ku.asin]])
        return products


class ReviewDataSet(Dataset):
    def __init__(self, x, users, product, split):
        self.x = x
        self.users = users
        self.product = product
        self.split = split
        self.text, self.users, self.products = self.split_feature_label()

    def split_feature_label(self):
        train_split = int(self.x.shape[0] * 0.8)
        valid_split = train_split - int(train_split * 0.2)
        if self.split == 'train':
            x = self.x[: valid_split]
            users, products = self.users[: valid_split], self.product[: valid_split]
        elif self.split == 'valid':
            x = self.x[valid_split: train_split]
            users, products = self.users[valid_split: train_split], self.product[valid_split: train_split]
        else:
            x = self.x[train_split:]
            users, products = self.users[train_split:], self.product[train_split:]
        return torch.tensor(x, dtype=torch.float), torch.tensor(users, dtype=torch.long), \
               torch.tensor(products, dtype=torch.long)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx, :]
        user = self.users[idx]
        product = self.products[idx]
        return {'text': text, 'user': user, 'product': product}


class Input:
    def __init__(self, reviews,
                 pretrained,
                 batch_size,
                 shuffle,
                 **param):
        self.reviews = reviews
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.info = self.get_info(**param)
        self.train_loader, self.valid_loader, self.test_loader = self.get_dataloader()

    def get_info(self, **param):
        '''
        :param param: if pretrained: param = {'feature_file', 'bert_vocab', 'max_seq_len', 'feature_dim'}
        :return: if not pretrained: param = {'min_threshold', 'feature_name', 'max_seq_len'}
        '''
        if self.pretrained:
            info = PreTrainedInfo(reviews=self.reviews,
                                  feature_file=param['feature_file'],
                                  feature_dim=param['feature_dim'],
                                  max_seq_len=param['max_seq_len'],
                                  bert_vocab=param['bert_vocab'])

        else:
            info = NonPreTrainedInfo(reviews=self.reviews,
                                     min_threshold=param['min_threshold'],
                                     feature_name=param['feature_name'],
                                     max_seq_len=param['max_seq_len'])
        return info

    def get_dataloader(self):
        train_dataset = ReviewDataSet(self.info.x, self.info.users, self.info.products, 'train')
        valid_dataset = ReviewDataSet(self.info.x, self.info.users, self.info.products, 'valid')
        test_dataset = ReviewDataSet(self.info.x, self.info.users, self.info.products, 'test')

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=5)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=5)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=5)

        return train_loader, valid_loader, test_loader


# if __name__ == '__main__':
#     reviews = get_reviews()
#     bert_model_vocab = '/home/leeyang/research/model/bert/vocab.txt'
#     feature_file = '/home/leeyang/research/model/feature_last.json'
#     review_input = Input(reviews=reviews, bert_model_vocab=bert_model_vocab,
#                          feature_file=feature_file, max_seq_len=3500, feature_dim=768)
    # texts, users = review_input.feature_label()




