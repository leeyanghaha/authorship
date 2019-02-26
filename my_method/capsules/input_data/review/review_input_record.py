import tensorflow as tf
from utils.vocabulary_utils import Vocabulary
import utils.key_utils as ku
from utils.data_utils import UserHelper, ReviewLoader, FeatureLoader
import os
import keras
import numpy as np
import keras



def _read_and_decode(split, max_ngram_len, feature='n-gram'):
    voca = Vocabulary(ku.voca_root)
    userhelper = UserHelper()
    reviews  = ReviewLoader(ku.Movie, product_num=50).get_data()

    users = userhelper.get_users(reviews)
    user2idx = userhelper.user2idx(users)
    if feature == 'n-gram':
        feature2idx = voca.character_n_gram_table(reviews, min_threshold=6)
    else:
        feature2idx = voca.word_table(reviews, min_threshold=5)
    print('--------------------feature2idx-----------------', len(feature2idx))
    feature_loader = FeatureLoader(user2idx=user2idx, max_ngram_len=max_ngram_len,
                                   ngram2idx=feature2idx)
    training_split = int(len(reviews) * 0.8)
    valid_split = training_split - int(training_split * 0.2)
    if split == 'train':
        X, Y = feature_loader.load_n_gram_idx_feature_label(reviews[: valid_split], )
    elif split == 'valid':
        X, Y = feature_loader.load_n_gram_idx_feature_label(reviews[: valid_split])
    else:
        X, Y = feature_loader.load_n_gram_idx_feature_label(reviews[training_split: ])
    # X, Y = tf.convert_to_tensor(X, dtype=tf.int32), tf.convert_to_tensor(Y, dtype=tf.int32)
    recons_Y = Y
    Y = keras.utils.to_categorical(Y, num_classes=len(user2idx))
    features = {'text': X, 'labels': Y, 'recons_labels': recons_Y}
    print('X.shape: ', X.shape)
    print('Y.shape: ', Y.shape)
    return features, len(user2idx), len(feature2idx), X.shape[0]


def _parse_function(text, label):
    return text, label


def inputs(batch_size, split, max_ngram_len, feature):
    with tf.name_scope('input'):
        features, num_classes, ngram_num, size = _read_and_decode(split, max_ngram_len, feature)
        features = tf.train.shuffle_batch(
          features, enqueue_many=True, allow_smaller_final_batch=True,
          batch_size=batch_size,
          num_threads=3,
          capacity=500 + 3 * 32,
          # Ensures a minimum amount of shuffling of examples.
          min_after_dequeue=500)
        features['num_classes'] = num_classes
        features['ngram_num'] = ngram_num
        features['size'] = size
        features['max_ngram_len'] = max_ngram_len
        return features



if __name__ == '__main__':
    pass
    # with tf.device('/gpu:2') as f:
    #     input(64, 'train', 3500)