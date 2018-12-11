import utils.key_utils as ku
import utils.function_utils as fu
import requests
import os
import json
from collections import Counter
import numpy as np

# files = fu.listchildren('/home/nfs/yangl/research/authorship/data/product/')
# meta_file = '/home/nfs/yangl/research/authorship/data/meta'
# #
# for file in files:
#     asin_set = {}
#     path = os.path.join('/home/nfs/yangl/research/authorship/data/product/', file)
#     reviews = fu.load_array(file)
#     for review in reviews:
#         asin = review[ku.asin]
#         if asin in asin_set:
#             asin_set[asin].append(review[ku.reviewer_ID])
#         else:
#             asin_set.update({asin:[review[ku.reviewer_ID]]})
#     with open(meta_file, 'a') as f:
#         f.write(json.dumps(asin_set) + '\n')


# users = []
# for key, item in asin_set.items():
#     users.extend(item)
#
# users = set(users)
# print('{}/{}'.format(len(users), 10 * 100000))

# file = '/home/nfs/yangl/research/authorship/data/meta'
new = '/home/nfs/yangl/research/authorship/data/product2user.json'
# all = {}
#
# arr = fu.load_array(file)
# #
# for i in arr:
#     for asin in i:
#         if asin in all:
#             all[asin].extend(i[asin])
#         else:
#             all.update({asin: i[asin]})
# #
# #
# with open(new, 'a') as f:
#     f.write(json.dumps(all))

# total = 0
counter = Counter()

user2product = {}
item_list = []
with open(new) as f:
    line = f.readlines()
    line = json.loads(line[0])
    for asin, users in line.items():
        counter.update(users)

condidate_users = counter.most_common(1000)

condidate_users_file = 'home/nfs/yangl/research/authorship/data/condidate.json'

with open(condidate_users_file, 'a') as f:
    for user in condidate_users:
        f.write(user[0] + '\n')


with open(new) as f:
    line = f.readlines()
    line = json.loads(line[0])
    for asin, users in line.items():
        for user in users:
            if user in condidate_users:
                if user in user2product:
                    user2product[user].append(asin)
                else:
                    user2product.update({user: [asin]})
user2product_file = '/home/nfs/yangl/research/authorship/data/user2product.json'
with open(user2product_file, 'a') as f:
    f.write(json.dumps(user2product_file))
#
# rand_seed = np.random.randint(0, len(item_list), 40)
# users_list = []
# for i in rand_seed:
#     users = item_list[i][1]
#     print(users)
#     users_list.extend(users)
#
#
# print(len(set(users_list)))


# condidate_products_file = '/home/nfs/yangl/research/authorship/data/condidate_products'
# condidate_products = []
# with open(new) as f:
#     line = f.readlines()
#     line = json.loads(line[0])
#     for asin, users_list in line.items():
#         for user in users_list:
#             if user in users:
#                 print('1')
#                 condidate_products.append(asin)
#
# with open(condidate_products_file, 'a') as f:
#     for product in condidate_products:
#         f.write(product + '\n')

