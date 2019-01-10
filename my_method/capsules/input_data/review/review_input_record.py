import tensorflow as tf
from utils.vocabulary_utils import Vocabulary
import utils.key_utils as ku
from utils.data_utils import UserHelper, ReviewLoader, FeatureLoader
import os
import keras
import numpy as np

def _read_and_decode(split, max_ngram_len):
    voca = Vocabulary(ku.voca_root)
    userhelper = UserHelper()
    reviews  = ReviewLoader(ku.Movie, product_num=100).get_data()

    users = userhelper.get_users(reviews)
    user2idx = userhelper.user2idx(users)

    ngram2idx = voca.character_n_gram_table(reviews, min_threshold=2)
    feature_loader = FeatureLoader(user2idx=user2idx, max_ngram_len=max_ngram_len,
                                   ngram2idx=ngram2idx)

    training_split = int(len(reviews) * 0.8)
    if split == 'train':
        X, Y = feature_loader.load_n_gram_idx_feature_label(reviews[: training_split], )
    else:
        X, Y = feature_loader.load_n_gram_idx_feature_label(reviews[training_split: ])
    # X, Y = tf.convert_to_tensor(X, dtype=tf.int32), tf.convert_to_tensor(Y, dtype=tf.int32)
    features = {'text': X, 'label': Y, 'num_classes': len(user2idx), 'ngram_num': len(ngram2idx)}
    return features


def _parse_function(text, label):
    return text, label


def input(batch_size, split, max_ngram_len):
    with tf.name_scope('input'):
        data = _read_and_decode(split, max_ngram_len)
        X, Y = data['text'], data['label']
        num_classes, ngram_num= data['num_classes'], data['ngram_num']
        dataset = tf.data.Dataset().from_tensor_slices((X, Y))
        dataset = dataset.shuffle(buffer_size=1000).batch(batch_size=batch_size)
    return {'dataset': dataset, 'num_classes': num_classes, 'ngram_num': ngram_num}



if __name__ == '__main__':
    with tf.device('/gpu:2') as f:
        input(64, 'train', 3500)