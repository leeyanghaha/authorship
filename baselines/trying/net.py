import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TextCNN(nn.Module):
    def __init__(self, max_len, embedding_dim, vocab_size, user_num):
        super(TextCNN, self).__init__()
        self.user_num = user_num
        self.ngram_embeds_layer = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv2d(1, 300, (3, embedding_dim))
        self.conv2 = nn.Conv2d(1, 300, (5, embedding_dim))
        self.conv3 = nn.Conv2d(1, 300, (7, embedding_dim))
        self.bn = nn.BatchNorm2d(300)
        self.bn1 = nn.BatchNorm1d(300)
        self.fc1 = nn.Linear(3 * 300, 300)
        self.fc2 = nn.Linear(300, self.user_num)
        self.softmax = nn.Softmax(dim=-1)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i
        return num_features

    def forward(self, x):
        ngram_embeds = self.ngram_embeds_layer(x)
        conv1 = self.conv1(ngram_embeds.unsqueeze_(1))
        bn1 = self.bn(conv1)
        max_pool1 = F.max_pool2d(F.relu(bn1), (conv1.shape[2], 1))
        max_pool1 = max_pool1.view(-1, self.num_flat_features(max_pool1))

        conv2 = self.conv2(ngram_embeds)
        bn2 = self.bn(conv2)
        max_pool2 = F.max_pool2d(F.relu(bn2), (conv2.shape[2], 1))
        max_pool2 = max_pool2.view(-1, self.num_flat_features(max_pool2))

        conv3 = self.conv3(ngram_embeds)
        bn3 = self.bn(conv3)
        max_pool3 = F.max_pool2d(F.relu(bn3), (conv3.shape[2], 1))
        max_pool3 = max_pool3.view(-1, self.num_flat_features(max_pool3))
        #
        flatten = torch.cat((max_pool1, max_pool2, max_pool3), 1)
        flatten = F.dropout2d(flatten, p=0.5)
        fc1 = self.fc1(flatten)
        fc1 = self.bn1(fc1)
        fc1 = F.dropout2d(fc1, p=0.5)
        fc2 = self.fc2(fc1)
        # out = self.softmax(fc2)
        return fc2


class TextProdCNN(nn.Module):
    def __init__(self, max_len, embedding_dim, vocab_size, products_num, user_num):
        super(TextProdCNN, self).__init__()
        self.user_num = user_num
        self.ngram_embeds_layer = nn.Embedding(vocab_size, embedding_dim)
        self.product_embeds_layer = nn.Embedding(products_num, embedding_dim)
        self.conv1 = nn.Conv2d(1, 300, (3, embedding_dim))
        self.conv2 = nn.Conv2d(1, 300, (5, embedding_dim))
        self.conv3 = nn.Conv2d(1, 300, (7, embedding_dim))
        self.fc1 = nn.Linear(3 * 300, self.user_num)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i
        return num_features

    def forward(self, x):
        ngram_embeds = self.ngram_embeds_layer(x)
        # add one channel for depth: shape(batch_size, 1, max_len, embedding_dim)
        ngram_embeds = ngram_embeds.unsqueeze_(1)
        # print('ngram.shape', ngram_embeds.shape)
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

        fc = self.fc1(flatten)
        return fc