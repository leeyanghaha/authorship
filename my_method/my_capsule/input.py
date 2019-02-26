from torch.utils.data import Dataset
import torch
import utils.key_utils as ku
from utils.data_utils import FeatureLoader, UserHelper
from utils.vocabulary_utils import Vocabulary
import numpy as np


class ReviewInfo:
    def __init__(self, reviews, min_threshold=6, feature_name='n-gram',
                 max_len=3500):
        self.vocab = Vocabulary(ku.voca_root)
        self.reviews = reviews
        self.feature = feature_name
        self.min_threshold = min_threshold
        self.max_len = max_len
        self.user2idx = self.user2idx()
        self.feature2idx = self.feature2idx(min_threshold)
        self.vocab_size = len(self.feature2idx)
        self.num_classes = len(self.user2idx)
        self.x, self.y = self.feature_label()
        self.fake_x, self.fake_y = self.fake_feature_label()

    def user2idx(self):
        userhelper = UserHelper()
        users = userhelper.get_users(self.reviews)
        return userhelper.user2idx(users)

    def feature2idx(self, min_threshold):
        assert self.feature == 'n-gram' or self.feature == 'word'
        if self.feature == 'n-gram':
            feature2idx = self.vocab.character_n_gram_table(self.reviews,
                                                            min_threshold=min_threshold)
        else:
            feature2idx = self.vocab.word_table(self.reviews, min_threshold=0)
        return feature2idx

    def feature_label(self):
        data_params = {'max_ngram_len': self.max_len, 'user2idx': self.user2idx, 'ngram2idx': self.feature2idx}
        feature_loader = FeatureLoader(**data_params)
        x, y = feature_loader.load_n_gram_idx_feature_label(self.reviews)
        return x, y

    def fake_feature_label(self):
        num_samples = len(self.reviews)
        x = np.random.randint(0, self.vocab_size, (num_samples, self.max_len))
        y = np.random.randint(0, self.num_classes, (num_samples, ))
        return x, y


class ReviewDataSet(Dataset):
    def __init__(self, x, y, split):
        self.x = x
        self.y = y
        self.split = split
        self.text, self.label = self.split_feature_label()

    def split_feature_label(self):
        train_split = int(self.x.shape[0] * 0.8)
        valid_split = train_split - int(train_split * 0.2)
        if self.split == 'train':
            x, y = self.x[: valid_split, :], self.y[: valid_split]
        elif self.split == 'valid':
            x, y = self.x[valid_split: train_split, :], self.y[valid_split: train_split]
        else:
            x, y = self.x[train_split:, :], self.y[train_split:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx, :]
        label = self.label[idx]
        return {'text': text, 'label': label}







