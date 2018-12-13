import utils.function_utils as fu
import utils.key_utils as ku
import utils.data_utils as du
import numpy as np
import utils.photo_utils as phu
import os
import sklearn.utils as sku
import json
from collections import Counter
import copy

domain = ku.Movie

product2user = fu.load_array(os.path.join(ku.index_root, domain, 'product2user.json'))[0]
user2product = fu.load_array(os.path.join(ku.index_root, domain, 'user2product.json'))[0]
all_products = fu.listchildren(os.path.join(ku.product_root, domain), concat=False)

product_num = 50
iter_num = 1000

def get_product_reviews(product):
    path = os.path.join(ku.product_root, domain, product)
    reviews = fu.load_array(path)
    return reviews


def sample_reviews_for_user(user, num):
    path = os.path.join(ku.user_root, domain, user)
    user_reviews = fu.load_array(path)[: num]
    return user_reviews


def remove_rare_users(result, user_counter, threshold):
    removing_users = set()
    for user, count in dict(user_counter).items():
        if count < threshold:
            removing_users.add(user)
    for pro, reviews in result.items():
        temp = []
        for review in reviews:
            if review[ku.reviewer_ID] not in removing_users:
                temp.append(review)
            result[pro] = temp
    return result


def check_efficiency(result):
    reviews = []
    for i in result:
        reviews.extend(result[i])
    users = userhelper.get_users(reviews)
    len_users = len(set(list(users)))
    len_reviews = len(reviews)
    print('每条product 有 {:.2f} 条 reviews '.format(len_reviews / product_num))
    print('users num', len_users)
    print('每个user 有 {:.2f} reviews'.format(len_reviews / len_users))
    print('products num: ', product_num)
    print('共有 {} 条 reviews.'.format(len_reviews))
    return reviews


def get_data(seed_product):
    candidate_products = set(all_products)
    product = seed_product
    candidate_products.remove(product)
    product_counter = Counter()
    user_counter = Counter()
    result = {}
    for i in range(product_num):

        next_product, product_counter_new, user_counter_new, result_new = iteration(product, product_counter, user_counter,
                                                                  user2product, result, candidate_products)
        if next_product != '':
            candidate_products.remove(next_product)
            product = next_product
            product_counter = product_counter_new
            user_counter = user_counter_new
            result = result_new
        else:
            continue
    result = remove_rare_users(result, user_counter, threshold=12)
    reviews = check_efficiency(result)
    return reviews


def iteration(product, product_counter, user_counter, user2product,
              result, candidate_products):
    reviews = get_product_reviews(product)
    users = userhelper.get_users(reviews)
    for user in users:
        product_counter.update(user2product[user])
    user_counter.update(users)
    result.update({product: reviews})
    most_product = product_counter.most_common()
    next_product = ''
    for p in most_product:
        p = p[0]
        if p in candidate_products:
            next_product = p
            break
        else:
            continue
    return next_product, product_counter, user_counter, result


def f_product2user(reviews):
    file = os.path.join(ku.index_root, domain, 'product2user.json')
    res = {}
    for review in reviews:
        product = review[ku.asin]
        user = review[ku.reviewer_ID]
        if product in res:
            res[product].append(user)
        else:
            res.update({product: [user]})
    with open(file, 'a') as f:
        f.write(json.dumps(res))


def f_user2product(reviews):
    file = os.path.join(ku.index_root, domain, 'user2product.json')

    res = {}
    for review in reviews:
        product = review[ku.asin]
        user = review[ku.reviewer_ID]
        if user in res:
            res[user].append(product)
        else:
            res.update({user: [product]})
    with open(file, 'a') as f:
        f.write(json.dumps(res))


def extract_product(reviews):
    for review in reviews:
        product = review[ku.asin]
        path = os.path.join(ku.product_root, domain, product)
        with open(path, 'a') as f:
            f.write(json.dumps(review) + '\n')


def get_reviews():
    files = fu.listchildren(os.path.join(ku.user_root, domain))
    reviews = []
    for file in files:
        reviews.extend(fu.load_array(file))
    return reviews






if __name__ == '__main__':
    files = fu.listchildren(os.path.join(ku.product_root, domain))
    seed_product = 'B003EYVXV4'
    result = get_data(seed_product)
