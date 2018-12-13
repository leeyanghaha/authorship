import utils.key_utils as ku
import utils.data_utils as du
from utils.vocabulary_utils import Vocabulary
import os
from models.text_cnn import TextCNN

from collections import Counter

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'


ngram_min_threshold = 2
max_ngram_len = 3500
data_type = ku.review
datahelper = du.DataHelper(data_type)
voca = Vocabulary(ku.voca_root, data_type)
userhelper = du.UserHelper(data_type)
dataloader = du.DataLoader(ku.review, ku.Movie)

reviews  = du.ReviewLoader(ku.Movie, product_num=50).get_data()

user_counter = Counter()
product_counter = Counter()
for review in reviews:
    user_counter.update([review[ku.reviewer_ID]])
    product_counter.update([review[ku.asin]])

print(reviews[:10])
# print('user counter: ', user_counter)
# print('product counter: ', product_counter)



