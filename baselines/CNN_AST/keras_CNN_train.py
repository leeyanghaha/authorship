import utils.key_utils as ku
from utils.data_utils import ReviewLoader, FeatureLoader, UserHelper
from utils.vocabulary_utils import Vocabulary
import os
import utils.function_utils as fu
from models.text_cnn import TextCNN

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


ngram_min_threshold = 6
max_ngram_len = 3500


voca = Vocabulary(ku.voca_root)
userhelper = UserHelper()

reviews  = ReviewLoader(ku.Movie, product_num=100).get_data()




users = userhelper.get_users(reviews)
user2idx = userhelper.user2idx(users)
ngram2idx = voca.character_n_gram_table(reviews, min_threshold=ngram_min_threshold)

data_params = {'max_ngram_len': max_ngram_len, 'user2idx': user2idx, 'ngram2idx': ngram2idx}
feature_loader = FeatureLoader(**data_params)


param = {'kernel_size': [3, 5, 7], 'batch_size': 64, 'epochs': 100, 'loss': 'categorical_crossentropy',
 'embedding_dim': 300, 'user_num': len(user2idx), 'max_ngram_len': max_ngram_len,  'feature_num':300 ,
         'vocab_size': len(ngram2idx)}
#
#
x, y = feature_loader.load_n_gram_idx_feature_label(reviews)


training_split = int(0.8 * x.shape[0])
training_x, training_y = x[:training_split, :], y[:training_split]
testing_x, testing_y = x[training_split:, ], y[training_split:]

model = TextCNN(**param)
model.fit(training_x, training_y)
model.save_weight(ku.CNN_AST_model)
model.load_weight(ku.CNN_AST_model)
res = model.evaluate(testing_x, testing_y)
testing_loss = res[0]
testing_acc = res[1]
print('testing_loss: {}, testing_acc: {}'.format(testing_loss, testing_acc))
