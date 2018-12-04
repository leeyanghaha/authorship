import utils.key_utils as ku
import utils.data_utils as du
from utils.vocabulary_utils import Vocabulary
import baselines.Syntax_CNN.model as syntax_cnn
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

max_user_num = 50
num_reviews_per_user = 50
ngram_min_threshold = 2
max_ngram_len = 600
max_pos_num = 10
max_words_num = 500
syntax_dim = 60
ngram_dim = 300


batch_size = 32
epoch = 100

data_type = ku.twitter
datahelper = du.DataHelper(data_type)
voca = Vocabulary(ku.voca_root, data_type)
userhelper = du.UserHelper(data_type)

if data_type == ku.review:
    dataloader = du.ReviewDataLoader(data_type, ku.Kindle, min_threshold=num_reviews_per_user
                           , num_reviews_per_user=num_reviews_per_user)
    reviews = dataloader.load_domain_reviews()
    users = userhelper.get_users(reviews, max_user_num)
    users = userhelper.sample_user(users, max_user_num)
    reviews = dataloader.load_users_data(users)
else:
    dataloader = du.TwitterDataLoader(data_type, num_reviews_per_num=num_reviews_per_user)
    users = dataloader.get_users(max_user_num)
    reviews = dataloader.load_users_data(users)

user2idx = userhelper.user2idx(users)
ngram2idx = voca.character_n_gram_table(reviews, min_threshold=ngram_min_threshold)
pos2idx = userhelper.pos2idx()
print('len reviews: ', len(reviews))


feature = dataloader.syntax_get_feature_label(reviews, user2idx, ngram2idx, pos2idx, max_ngram_len,
                                             max_pos_num, max_words_num)

pos_id, position_id, ngram_id, user_id = feature[ku.pos_id], feature[ku.pos_order_id], \
                                         feature[ku.ngram_id], feature[ku.user_id]

print('pos_id: ', pos_id.shape)
print('position_id: ', position_id.shape)
print('ngram_id: ', ngram_id.shape)
print('user_id: ', user_id.shape)

training_split = int(0.8 * ngram_id.shape[0])
training_ngram_id, testing_ngram_id = ngram_id[:training_split, :], ngram_id[training_split:, :]
training_pos_id, testing_pos_id = pos_id[:training_split, :], pos_id[training_split:, :]
training_position_id, testing_position_id = position_id[:training_split, :], position_id[training_split:, :]
training_x = [training_ngram_id, training_pos_id, training_position_id]
testing_x = [testing_ngram_id, testing_pos_id, testing_position_id]

training_y = user_id[:training_split]
testing_y = user_id[training_split:]


model = syntax_cnn.Net(max_words_num, syntax_dim, max_ngram_len, ngram_dim, len(ngram2idx),
                 len(pos2idx), len(user2idx), max_pos_num)


model.fit(training_x, training_y)
model.save_weight(ku.Syntax_CNN_model)
# model.load_weight(ku.CNN_AST_model)
res = model.evaluate(testing_x, testing_y)
testing_loss = res[0]
testing_acc = res[1]
print('testing_loss: {}, testing_acc: {}'.format(testing_loss, testing_acc))


if __name__ == '__main__':
    pass
