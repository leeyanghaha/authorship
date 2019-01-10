import tensorflow as tf
from utils.vocabulary_utils import Vocabulary
import utils.key_utils as ku
from utils.data_utils import UserHelper, ReviewLoader, FeatureLoader
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


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
        X, Y = feature_loader.load_n_gram_idx_feature_label(reviews[: training_split])
    else:
        X, Y = feature_loader.load_n_gram_idx_feature_label(reviews[training_split: ])
    # X, Y = tf.convert_to_tensor(X, dtype=tf.int32), tf.convert_to_tensor(Y, dtype=tf.int32)
    features = {'text': X, 'label': Y}
    return features

features = _read_and_decode('train', 3500)
dataset = tf.data.Dataset().from_tensor_slices(features)
iterator = dataset.make_initializable_iterator()
init_op = iterator.initializer

next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(next_element))






