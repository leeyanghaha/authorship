import os
import utils.function_utils as fu
import json
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils.data_utils import FeatureLoader, UserHelper, DataHelper
import torch
from utils.vocabulary_utils import Vocabulary
import utils.key_utils as ku
from scipy import sparse

root_dir = '/home/leeyang/research/data/twitter'
all_user_tweets = os.path.join(root_dir, 'select_user')
tweets_file = os.path.join(root_dir, 'training_bert')

# def get_user_tweets(num_tweets):
#     res = []
#     lines = fu.load_array(all_user_tweets)
#     for line in lines:
#         tweets = []
#         i = 0
#         for text in line['tweets']:
#             i += 1
#             text = fu.text_normalization(text)
#             tweets.append(text)
#             if i % num_tweets == 0:
#                 res.append({'job': line['job'], 'tweets': fu.text_normalization(' '.join(tweets).lower())})
#                 tweets = []
#     return res


def get_user_tweets():
    return fu.load_array(tweets_file)


class TwitterInfo:
    def __init__(self, tweets, feature_name, max_seq_len):
        self.max_seq_len = max_seq_len
        self.feature_name = feature_name
        self.tweets = tweets
        self.vocab = Vocabulary(ku.voca_root)
        self.feature_dim = 300
        self.feature2idx = self.feature2idx()
        self.job2idx = self.job2idx()
        self.num_class = len(self.job2idx)
        self.x, self.jobs = self.feature_label()

    def job2idx(self):
        job_num = -1
        res = {}
        for tweet in self.tweets:
            job = tweet['job']
            if job not in res:
                res.update({job: job_num + 1})
                job_num += 1
        return res

    def feature2idx(self):
        if self.feature_name == 'n-gram':
            feature2idx = self.vocab.character_n_gram_table(self.tweets, min_threshold=6)
        else:
            feature2idx = self.vocab.word_table(self.tweets, min_threshold=6)
        return feature2idx

    def feature_label(self):
        data_params = {'max_ngram_len': self.max_seq_len, 'user2idx': self.job2idx, 'ngram2idx': self.feature2idx}
        feature_loader = FeatureLoader(**data_params)
        # x, y = feature_loader.load_n_gram_idx_feature_label(self.tweets, padding=False)
        x, y = feature_loader.load_n_gram_binary_feature_label(self.tweets)
        return x, y

class TwitterDataset(Dataset):
    def __init__(self, x, jobs, split):
        self.x = x
        self.jobs = jobs
        self.split = split
        self.text, self.jobs = self.split_feature_label()

    def split_feature_label(self):
        train_split = int(self.x.shape[0] * 0.8)
        valid_split = train_split - int(train_split * 0.2)
        if self.split == 'train':
            x, jobs = self.x[: valid_split], self.jobs[: valid_split]
        elif self.split == 'valid':
            x, jobs = self.x[valid_split: train_split], self.jobs[valid_split: train_split]
        else:
            x = self.x[train_split:], self.jobs[train_split:]
        x = torch.tensor(x, dtype=torch.float) if len(x.shape) > 2 else torch.tensor(x, dtype=torch.long)
        return x, torch.tensor(jobs, dtype=torch.long)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx, :]
        job = self.jobs[idx]
        return {'text': text, 'job': job}


class Input:
    def __init__(self, tweets,
                 batch_size,
                 shuffle,
                 **param):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.info = TwitterInfo(tweets, **param)
        self.train_loader, self.valid_loader, self.test_loader = self.get_dataloader()

    def get_dataloader(self):
        train_dataset = TwitterDataset(self.info.x, self.info.jobs, 'train')
        valid_dataset = TwitterDataset(self.info.x, self.info.jobs, 'valid')
        test_dataset = TwitterDataset(self.info.x, self.info.jobs, 'test')

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=5)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=5)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=5)

        return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    tweets = get_user_tweets()
    fu.dump_array(tweets, '/home/leeyang/research/data/twitter/training_bert')


