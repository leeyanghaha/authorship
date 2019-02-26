import utils.key_utils as ku
from utils.data_utils import ReviewLoader, UserHelper, DataHelper
import baselines.only_product.data_process as dp
from models.only_product_model import OnlyProduct
import os
import numpy as np
import baselines.product_embeds.embeds_model as em

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

userhelper = UserHelper()
datahelper = DataHelper()
reviews  = ReviewLoader(ku.Movie, product_num=100).get_data()

users = userhelper.get_users(reviews)
user2idx = userhelper.user2idx(users)

products = datahelper.get_products(reviews)
product2idx = datahelper.product2idx(products)
products_id = datahelper.load_products_id(products, product2idx)

embedding_dim = 300
node2vec_param = {'path_length': 10, 'num_paths': 10, 'dim': embedding_dim,
                  'workers': 8, 'p': 0.6, 'q': 0.5, 'dw': False}

# em.node2vec_train(reviews, product2idx, node2vec_param)
# product_embedding = em.load_node2vec_embedding(ku.products_embeds, len(product2idx), embedding_dim)


x, y = dp.load_feature_label(reviews, products_id)
#
params = {'batch_size': 16, 'embedding_dim': embedding_dim, 'user_num': len(user2idx),
          'product_num': len(product2idx), 'pre_train_embeds': None}

training_split = int(0.8 * x.shape[0])
training_x, training_y = x[:training_split], y[:training_split]
testing_x, testing_y = x[training_split: ], y[training_split: ]

model = OnlyProduct(**params)
model.fit(training_x, training_y)
res = model.evaluate(testing_x, testing_y)
print(testing_y)
# print(model.predict(testing_x))
testing_loss = res[0]
testing_acc = res[1]
print('testing_loss: {}, testing_acc: {}'.format(testing_loss, testing_acc))



