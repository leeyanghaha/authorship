import torch.nn as nn
import utils.function_utils as fu
import collections
import logging
import json

import utils.key_utils as ku
import torch
from torch.utils.data import TensorDataset, DataLoader

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

file = '/home/leeyang/research/model/CD_feature.json'
arr = fu.load_array(file)
#
last_layer = '/home/leeyang/research/model/CD_feature_last_300.json'
# second_layer = '/home/leeyang/research/model/feature_second.json'
#
last_dict = {}
second_dict = {}

for line in arr:
    for feature in line['features']:
        token, layers = feature['token'], feature['layers']
        for layer in layers:
            if layer['index'] == -1:
                print(layer['values'][: 300])
                last_dict.update({token: layer['values'][:300]})
            # elif layer['index'] == -2:
            #     second_dict.update({token: layer['values']})
#
# print('last_dict', last_dict[0])
# fu.dump_array(last_dict, last_layer)
# # fu.dump_array(second_dict, second_layer)
#
with open(last_layer, 'a') as f:
    json.dump(last_dict, f)
#
# with open(second_layer, 'a')as f:
#     json.dump(second_dict, f)

# last_layer_300 = '/home/leeyang/research/model/feature_last_300.json'
#
# second_layer_300 = '/home/leeyang/research/model/feature_second_300.json'
#
# last_300 = {}
# second_300 = {}
#
# for key, value in last_dict.items():
#     last_300.update({key: value[: 300]})

# for key, value in second_dict.items():
#     second_300.update({key: value[: 300]})

# with open(last_layer_300, 'a') as f:
#     json.dump(last_300, f)
#
# with open(second_layer_300, 'a') as f:
#     json.dump(second_300, f)

