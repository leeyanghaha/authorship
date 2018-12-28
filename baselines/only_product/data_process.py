from utils.data_utils import FeatureLoader, UserHelper, DataHelper
import utils.key_utils as ku
import numpy as np
from collections import Counter
from scipy import sparse


userhelper = UserHelper()
datahelper = DataHelper()
feature_loader = FeatureLoader()



def get_users(reviews):
    users = userhelper.get_users(reviews)
    user2idx = userhelper.user2idx(users)
    users_id = []
    for review in reviews:
        users_id.append(user2idx[review[ku.reviewer_ID]])
    return np.array(users_id)
#
#
# def get_products_id(reviews):
#     products = datahelper.get_products(reviews)
#     product2idx = datahelper.product2idx(products)
#     products_id = datahelper.load_products_id(products, product2idx)
#     return products_id, len(product2idx)


def load_feature_label(reviews, products_id):
    y = get_users(reviews)
    # products_id, num_class = get_products_id(reviews)
    # x = feature_loader.one_hot_encoding(products_id, num_class)
    # if not to_dense:
    #     x = sparse.csc_matrix(x)
    return products_id, y


