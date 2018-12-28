from models.text_cnn_product import TextCNNPro, SelfRegulationPro
from models.text_cnn import TextCNN
import utils.data_utils as du
import utils.key_utils as ku
from utils.vocabulary_utils import Vocabulary
import my_method.data_process as dp
import os
from utils.data_utils import ReviewLoader, FeatureLoader, UserHelper, DataHelper
import baselines.product_embeds.embeds_model as em


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

max_ngram_len = 3500
embedding_dim = 300


datahelper = DataHelper()
voca = Vocabulary(ku.voca_root)
userhelper = UserHelper()


review_loader  = ReviewLoader(ku.Movie, product_num=100)
reviews = review_loader.get_data()
users = userhelper.get_users(reviews)

user2idx = userhelper.user2idx(users)
ngram2idx = voca.character_n_gram_table(reviews, min_threshold=6)

products = datahelper.get_products(reviews)
product2idx = datahelper.product2idx(products)
products_id = review_loader.load_products_id(products, product2idx)

product_embedding = em.load_node2vec_embedding(ku.products_embeds, len(product2idx), embedding_dim)

param = {'kernel_size': [3, 5, 7], 'batch_size': 64, 'epochs': 100,
 'embedding_dim': embedding_dim, 'user_num': len(user2idx), 'max_ngram_len': max_ngram_len,  'feature_num':300 ,
         'vocab_size': len(ngram2idx), 'product_embedding_dim': 3 * 100, 'product_num': len(products),
         'pre_trained_embeds': product_embedding, 'method': 1}

x, y = dp.load_ngram_feature_label(reviews, user2idx, ngram2idx, max_ngram_len=max_ngram_len)


training_split = int(0.8 * x.shape[0])
training_x, training_y = [x[:training_split, :], products_id[:training_split]], y[:training_split]
testing_x, testing_y = [x[training_split:, ], products_id[training_split:]], y[training_split:]

print('this is method {}'.format(param['method']))
model = TextCNNPro(**param)
model.fit(training_x, training_y)
model.save_weight(ku.CNN_AST_model)
# # model.load_weight(ku.CNN_AST_model)
res = model.evaluate(testing_x, testing_y)
testing_loss = res[0]
testing_acc = res[1]
print('testing_loss: {}, testing_acc: {}'.format(testing_loss, testing_acc))




