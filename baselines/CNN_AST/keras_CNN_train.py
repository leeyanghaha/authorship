import utils.key_utils as ku
import utils.data_utils as du
from utils.vocabulary_utils import Vocabulary
import os
import baselines.CNN_AST.data_process as dp
from models.text_cnn import TextCNN

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


ngram_min_threshold = 2
max_ngram_len = 3500
data_type = ku.review

voca = Vocabulary(ku.voca_root)
userhelper = du.UserHelper()

reviews  = du.ReviewLoader(ku.Movie, product_num=50).get_data()

users = userhelper.get_users(reviews)
#
#
user2idx = userhelper.user2idx(users)
ngram2idx = voca.character_n_gram_table(reviews, min_threshold=3)
# #
# #
# #
param = {'kernel_size': [4, 7, 9], 'batch_size': 32, 'epochs': 100, 'loss': 'categorical_crossentropy',
 'embedding_dim': 300, 'user_num': len(user2idx), 'max_ngram_len': max_ngram_len,  'feature_num':300 ,
         'vocab_size': len(ngram2idx)}
#
#
x, y = dp.load_ngram_idx_feature_label(reviews, user2idx, ngram2idx, max_ngram_len)

#
training_split = int(0.8 * x.shape[0])
training_x, training_y = x[:training_split, :], y[:training_split]
testing_x, testing_y = x[training_split:, ], y[training_split:]
print('training.shape: ', training_x.shape, training_y.shape)
print('testing shape: ', testing_x.shape, testing_y.shape)
model = TextCNN(**param)
model.fit(training_x, training_y)
model.save_weight(ku.CNN_AST_model)
# model.load_weight(ku.CNN_AST_model)
res = model.evaluate(testing_x, testing_y)
testing_loss = res[0]
testing_acc = res[1]
print('testing_loss: {}, testing_acc: {}'.format(testing_loss, testing_acc))
