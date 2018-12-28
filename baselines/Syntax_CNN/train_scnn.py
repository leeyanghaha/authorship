import utils.key_utils as ku
from utils.data_utils import ReviewLoader, UserHelper, DataHelper, FeatureLoader
from utils.vocabulary_utils import Vocabulary
import baselines.Syntax_CNN.model as syntax_cnn
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


ngram_min_threshold = 5
max_pos_num = 10
max_words_num = 500

batch_size = 32
epoch = 100


voca = Vocabulary(ku.voca_root)
userhelper = UserHelper()

reviews  = ReviewLoader(ku.Movie, product_num=50).get_data()

users = userhelper.get_users(reviews)
#
#
user2idx = userhelper.user2idx(users)
ngram2idx = voca.character_n_gram_table(reviews, min_threshold=ngram_min_threshold)
pos2idx = userhelper.pos2idx()


data_params = {ku.max_ngram_len: 600, ku.max_pos_num: max_pos_num,
               ku.max_words_num: max_words_num, ku.user2idx: user2idx,
               ku.ngram2idx: ngram2idx, ku.pos2idx: pos2idx,
               }

net_params = {ku.max_words_num: max_words_num, 'syntax_dim': 60, 'ngram_dim': 300,
              'pos_type_num': len(pos2idx), 'out_dim': len(user2idx),
              ku.max_pos_num: max_pos_num, 'vocab_size': len(ngram2idx),
              'batch_size': 32, 'filters': 300, 'kernel_size': 3,
              'loss': 'categorical_crossentropy'}

feature_loader = FeatureLoader(**data_params)
feature = feature_loader.syntax_cnn_feature_label(reviews)

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


model = syntax_cnn.Net(**net_params)


model.fit(training_x, training_y)
model.save_weight(ku.Syntax_CNN_model)
# model.load_weight(ku.CNN_AST_model)
res = model.evaluate(testing_x, testing_y)
testing_loss = res[0]
testing_acc = res[1]
print('testing_loss: {}, testing_acc: {}'.format(testing_loss, testing_acc))


if __name__ == '__main__':
    pass
