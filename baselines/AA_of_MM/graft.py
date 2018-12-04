import utils.function_utils as fu
import utils.key_utils as ku
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

from models.n_grams import Ngram
from baselines.LDAH_S import data
import utils.review_data_utils as du
import lda
import numpy as np


max_user_num = 10

dataloader = data.dataloader
datahelper = du.DataHelper()
userhelper = du.UserHelper()

reviews = dataloader.load_domain_reviews()
users = datahelper.get_users(reviews)
users = userhelper.sample_user(users, max_user_num)
user2idx = du.DataHelper().user2idx(users)
print(len(user2idx))
reviews_ordered_by_users = data.get_reviews(users)

training_text = []

for user, review in reviews_ordered_by_users.items():
    training_text.extend(review['train'])

lda_ = lda.LDA(n_topics=10)
stopwords = fu.get_stopwords()
print(stopwords)
counter = CountVectorizer(stop_words=stopwords)
x = counter.fit_transform(training_text)
lda_.fit(x)
topic_word = lda_.topic_word_
vocab = counter.get_feature_names()
n_top_words = 10

for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

