from torch.utils.data import Dataset
import torch
import utils.key_utils as ku
from utils.data_utils import FeatureLoader, UserHelper
from utils.vocabulary_utils import Vocabulary
import os


class ReviewDataSet(Dataset):
    def __init__(self, reviews, split, feature='n-gram', min_threshold=6, max_len=3500):
        self.vocab = Vocabulary(ku.voca_root)
        self.reviews = reviews
        self.feature = feature
        self.min_threshold = min_threshold
        self.max_len = max_len
        self.user2idx = self.user2idx()
        self.text, self.label, self.vocab_size = self.load_feature_label(split)

    def get_users(self):
        userhelper = UserHelper()
        return userhelper.get_users(self.reviews)

    def user2idx(self):
        userhelper = UserHelper()
        users = userhelper.get_users(self.reviews)
        return userhelper.user2idx(users)

    def feature2idx(self):
        assert self.feature == 'n-gram' or self.feature == 'word'
        if self.feature == 'n-gram':
            feature2idx = self.vocab.character_n_gram_table(self.reviews,
                                                            min_threshold=self.min_threshold)
        else:
            feature2idx = self.vocab.word_table(self.reviews, min_threshold=self.min_threshold)
        return feature2idx

    def load_feature_label(self, split):
        feature2idx = self.feature2idx()
        data_params = {'max_ngram_len': self.max_len, 'user2idx': self.user2idx, 'ngram2idx': feature2idx}
        feature_loader = FeatureLoader(**data_params)
        x, y = feature_loader.load_n_gram_idx_feature_label(self.reviews)
        train_split = int(x.shape[0] * 0.8)
        valid_split = train_split - int(train_split * 0.2)
        if split == 'train':
            x, y = x[: valid_split, :], y[: valid_split]
        elif split == 'valid':
            x, y = x[valid_split: train_split, :], y[valid_split: train_split]
        else:
            x, y = x[train_split: , :] , y[train_split:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long), len(feature2idx)


    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx, :]
        label = self.label[idx]
        return {'text': text, 'label': label}

