import utils.review_data_utils as du
import utils.key_utils as ku
import utils.function_utils as fu
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import math
import os

num_reviews_per_user = 100

dataloader = du.DataLoader(ku.user_root, ku.Kindle, min_threshold=100
                           , num_reviews_per_user=num_reviews_per_user)
datahelper = du.DataHelper()


training_split = math.ceil(num_reviews_per_user * 0.8)
valid_split = math.ceil(training_split * 0.2)


params = {'n_components': 20, 'learning_method': 'online', 'learning_decay': 0.7,
          'topic_word_prior': 0.01,
          'learning_offset': 10, 'batch_size': 64, 'mean_change_tol': 0.001,
          'max_doc_update_iter': 500000, 'n_jobs': 10}

lda = LDA(**params)

def get_reviews(users):
    '''
    为每一个user划分出训练集，验证集和测试集
    :param users:
    :return:
    '''
    res = {}
    for user in users:
        user_all_reviews = dataloader._load_user_reviews(user)
        user_reviews_text = datahelper.get_text(user_all_reviews)[:num_reviews_per_user]
        train = user_reviews_text[: training_split - valid_split]
        valid = user_reviews_text[training_split - valid_split: training_split]
        test = user_reviews_text[training_split: ]
        res.update({user: {'train': train, 'valid': valid, 'test': test}})
    return res




stopwords = fu.get_stopwords()


def get_topic_distribution(texts):
    vectorizer = CountVectorizer(stop_words=stopwords, lowercase=True, min_df=3, max_df=10000)
    x = vectorizer.fit_transform(texts)
    print(x.shape)
    # x = x.toarray()
    lda_res = lda.fit_transform(x)
    # print('topic distribution: ')
    # present_topic_words(lda, vectorizer)
    return lda_res


def get_training_feature(reviews_ordered_by_users, user2idx):
    training_text = [0 for _ in range(len(user2idx))]
    for user, review in reviews_ordered_by_users.items():
        train = ' '.join(review['train'])
        training_text[user2idx[user]] = train
    training_feature = get_topic_distribution(training_text)
    return training_feature


def get_valid_feature(reviews_ordered_by_users, user2idx):
    target = []
    valid_text = []
    for user, review in reviews_ordered_by_users.items():
        valid = review['valid']
        valid_text.extend(valid)
        target.extend([user2idx[user] for _ in range(len(valid))])
    valid_feature = get_topic_distribution(valid_text)
    return valid_feature, target


def get_test_feature(reviews_ordered_by_users, user2idx):
    target = []
    test_text = []
    for user, review in reviews_ordered_by_users.items():
        test = review['test']
        test_text.extend(test)
        target.extend([user2idx[user] for _ in range(len(test))])
    test_feature = get_topic_distribution(test_text)
    return test_feature, target


def hellinger_distance(feature1, feature2):
    return np.sqrt(np.sum((np.sqrt(feature1) - np.sqrt(feature2)) ** 2)) / np.sqrt(2)


def predict(valid_x, training_x):
    res = []
    for i in range(valid_x.shape[0]):
        min = 1000000
        pred = -1
        for j in range(training_x.shape[0]):
            distance = hellinger_distance(valid_x[i, :], training_x[j, :])
            if distance < min:
                min = distance
                pred = j
        # print('min: ', min)
        print('pred： ' ,pred)
        res.append(pred)
    return res


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def present_topic_words(lda, counter):
    feature_name = counter.get_feature_names()
    topic_distribution = lda.components_
    for i in range(topic_distribution.shape[0]):
        print('topic {}'.format(i))
        print(np.array(feature_name)[np.argsort(topic_distribution[i, :])][-1:-(10+1):-1])



if __name__ == '__main__':
    pass
