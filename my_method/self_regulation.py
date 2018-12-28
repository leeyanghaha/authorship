from models.text_cnn_product import TextCNNPro, SelfRegulationPro
from models.text_cnn import TextCNN
import utils.data_utils as du
import utils.key_utils as ku
from utils.vocabulary_utils import Vocabulary
import my_method.data_process as dp
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

max_ngram_len = 3500

datahelper = du.DataHelper()
voca = Vocabulary(ku.voca_root)
userhelper = du.UserHelper()


review_loader  = du.ReviewLoader(ku.Movie, product_num=50)
reviews = review_loader.get_data()
users = userhelper.get_users(reviews)

user2idx = userhelper.user2idx(users)
ngram2idx = voca.character_n_gram_table(reviews, min_threshold=3)

products = datahelper.get_products(reviews)
product2idx = datahelper.product2idx(products)
products_id = review_loader.load_products_id(products, product2idx)


param = {'kernel_size': [4, 7, 9], 'batch_size': 32, 'epochs': 100,
 'embedding_dim': 300, 'user_num': len(user2idx), 'max_ngram_len': max_ngram_len,  'feature_num':300 ,
         'vocab_size': len(ngram2idx), 'product_embedding_dim': 3 * 100, 'product_num': len(products)}

x, y = dp.load_ngram_feature_label(reviews, user2idx, ngram2idx, max_ngram_len=max_ngram_len)


training_split = int(0.8 * x.shape[0])
text_train_x , product_train_x, train_y = x[:training_split, :], products_id[:training_split], y[:training_split]
text_test_x, product_test_x, test_y = x[training_split:, ], products_id[training_split:], y[training_split:]


model = SelfRegulationPro(**param)
model.fit(inputs = {'text_input': text_train_x, 'product_input': product_train_x},
          outputs={'d_': train_y, 'd_f': train_y, 'o_g': train_y, 'o_g_f': train_y},
          )
# model.save_weight(ku.CNN_AST_model)
# # model.load_weight(ku.CNN_AST_model)
res = model.evaluate(inputs={'text_input': text_test_x, 'product_input': product_test_x},
               y_true= test_y)

print('test acc: ', res)
# testing_loss = res[0]
# testing_acc = res[1]
# print('testing_loss: {}, testing_acc: {}'.format(testing_loss, testing_acc))



