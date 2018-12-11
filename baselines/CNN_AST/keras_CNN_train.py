import utils.key_utils as ku
import utils.data_utils as du
from utils.vocabulary_utils import Vocabulary
import os
from models.text_cnn import TextCNN

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


max_user_num = 50
num_reviews_per_user = 100
min_threshold = 200
ngram_min_threshold = 2
max_ngram_len = 3500
data_type = ku.review
datahelper = du.DataHelper(data_type)
voca = Vocabulary(ku.voca_root, data_type)
userhelper = du.UserHelper(data_type)


dataloader = du.ReviewDataLoader(data_type, ku.Kindle, min_threshold=min_threshold,
                                 num_reviews_per_user=num_reviews_per_user)
reviews = dataloader.load_domain_reviews()
users = userhelper.get_users(reviews, max_user_num)
users = userhelper.sample_user(users, max_user_num)
reviews = dataloader.load_users_data(users)

user2idx = userhelper.user2idx(users)
ngram2idx = voca.character_n_gram_table(reviews, min_threshold=3)

print('len user: ', len(user2idx))
print('len ngram: ', len(ngram2idx))
print('len reviews: ', len(reviews))


param = {'kernel_size': [4, 7, 9], 'batch_size': 32, 'epochs': 100, 'loss': 'categorical_crossentropy',
 'embedding_dim': 300, 'user_num': len(user2idx), 'max_ngram_len': max_ngram_len,  'feature_num':300 ,
         'vocab_size': len(ngram2idx)}


x, y = dataloader.load_n_gram_feature_label(reviews, ngram2idx, user2idx, max_ngram_len=max_ngram_len,
                                            binary=False)

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
