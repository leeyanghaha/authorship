import torch
import torch.nn as nn
import torch.optim as optim
import my_method.my_capsule.models.layers as layers
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, embedding_dim, user_num):
        super(TextCNN, self).__init__()
        self.user_num = user_num
        # self.ngram_embeds_layer = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv2d(1, 300, (3, embedding_dim))
        self.conv2 = nn.Conv2d(1, 300, (5, embedding_dim))
        self.conv3 = nn.Conv2d(1, 300, (7, embedding_dim))
        self.linear = nn.Linear(3 * 300, self.user_num)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i
        return num_features

    def forward(self, x):
        ngram_embeds = x.unsqueeze(1)
        conv1 = self.conv1(ngram_embeds)
        max_pool1 = F.max_pool2d(F.relu(conv1), (conv1.shape[2], 1))
        max_pool1 = max_pool1.view(-1, self.num_flat_features(max_pool1))

        conv2 = self.conv2(ngram_embeds)
        max_pool2 = F.max_pool2d(F.relu(conv2), (conv2.shape[2], 1))
        max_pool2 = max_pool2.view(-1, self.num_flat_features(max_pool2))

        conv3 = self.conv3(ngram_embeds)
        max_pool3 = F.max_pool2d(F.relu(conv3), (conv3.shape[2], 1))
        max_pool3 = max_pool3.view(-1, self.num_flat_features(max_pool3))

        flatten = torch.cat((max_pool1, max_pool2, max_pool3), 1)
        # print('flatten: ', type(flatten), flatten.size())
        # torch.save(flatten, '/home/leeyang/research/model/imi/feature_for_product/{}/{}.pt'.format(set_type, i))
        out = self.linear(flatten)
        return out


class PoolProduct(nn.Module):
    def __init__(self, num_classes):
        super(PoolProduct, self).__init__()
        self.linear = nn.Linear(900*2, num_classes)

    def forward(self, x):
        return self.linear(x)

