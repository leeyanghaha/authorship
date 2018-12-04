import utils.function_utils as fu
import utils.key_utils as ku
import os
import numpy as np
import sklearn.utils as sku

files = fu.listchildren(ku.twitter_data_root)
new = '/home/yangl/research/authorship/data/twitter'

def change_data(data):
    res = []
    for item in data:
        item['id'] = item['user']['id']
        item['id_str']= item['user']['id_str']
        res.append(item)
    return res

for file in files:
    with open(file) as f:
        data = fu.load_array(file)
        data = change_data(data)
        user = data[1]['user']['id_str']
        new_file = os.path.join(new, user)
        fu.dump_array(data, new_file)













