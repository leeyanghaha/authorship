import torch
import torch.nn as nn
import torch.optim as optim
import my_method.my_capsule.models.layers as layers
import torch.nn.functional as F


class BertCNN(nn.Module):
    def __init__(self, embedding_dim, user_num):
        super(BertCNN, self).__init__()
        self.user_num = user_num
        self.conv1 = nn.Conv2d(1, 600, (1, embedding_dim))
        self.conv2 = nn.Conv2d(1, 600, (2, embedding_dim))
        self.conv3 = nn.Conv2d(1, 600, (3, embedding_dim))
        self.linear = nn.Linear(3 * 600, user_num)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i
        return num_features

    def forward(self, x):
        ngram_embeds = x
        ngram_embeds = ngram_embeds.unsqueeze(1)

        conv1 = self.conv1(ngram_embeds)
        conv1 = self.dropout1(conv1)
        max_pool1 = F.max_pool2d(F.relu(conv1), (conv1.shape[2], 1))
        max_pool1 = max_pool1.view(-1, self.num_flat_features(max_pool1))

        conv2 = self.conv2(ngram_embeds)
        conv2 = self.dropout2(conv2)
        max_pool2 = F.max_pool2d(F.relu(conv2), (conv2.shape[2], 1))
        max_pool2 = max_pool2.view(-1, self.num_flat_features(max_pool2))

        conv3 = self.conv3(ngram_embeds)
        conv3 = self.dropout3(conv3)
        max_pool3 = F.max_pool2d(F.relu(conv3), (conv3.shape[2], 1))
        max_pool3 = max_pool3.view(-1, self.num_flat_features(max_pool3))

        flatten = torch.cat((max_pool1, max_pool2, max_pool3), 1)
        flatten = self.dropout1(flatten)
        out = self.linear(flatten)
        return out


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, user_num):
        super(TextCNN, self).__init__()
        self.user_num = user_num
        self.ngram_embeds_layer = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv2d(1, 300, (3, embedding_dim))
        self.conv2 = nn.Conv2d(1, 300, (5, embedding_dim))
        self.conv3 = nn.Conv2d(1, 300, (7, embedding_dim))
        self.linear = nn.Linear(3 * 300, user_num)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i
        return num_features

    def forward(self, x):
        ngram_embeds = self.ngram_embeds_layer(x)
        ngram_embeds = ngram_embeds.unsqueeze(1)

        conv1 = self.conv1(ngram_embeds)
        conv1 = self.dropout1(conv1)
        max_pool1 = F.max_pool2d(F.relu(conv1), (conv1.shape[2], 1))
        max_pool1 = max_pool1.view(-1, self.num_flat_features(max_pool1))

        conv2 = self.conv2(ngram_embeds)
        conv2 = self.dropout2(conv2)
        max_pool2 = F.max_pool2d(F.relu(conv2), (conv2.shape[2], 1))
        max_pool2 = max_pool2.view(-1, self.num_flat_features(max_pool2))

        conv3 = self.conv3(ngram_embeds)
        conv3 = self.dropout3(conv3)
        max_pool3 = F.max_pool2d(F.relu(conv3), (conv3.shape[2], 1))
        max_pool3 = max_pool3.view(-1, self.num_flat_features(max_pool3))

        flatten = torch.cat((max_pool1, max_pool2, max_pool3), 1)
        flatten = self.dropout1(flatten)
        out = self.linear(flatten)
        return out


class BertProduct(nn.Module):
    def __init__(self, embedding_dim, user_num, product_num):
        super(BertProduct, self).__init__()
        self.product_embeds_layer = nn.Embedding(product_num, 300)
        self.conv1 = nn.Conv2d(1, 600, (1, embedding_dim))
        self.conv2 = nn.Conv2d(1, 600, (2, embedding_dim))
        self.conv3 = nn.Conv2d(1, 600, (3, embedding_dim))
        self.product_conv = nn.Conv1d(1, 600, 5, padding=2)
        self.linear = nn.Linear(4 * 600, user_num)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i
        return num_features

    def forward(self, x, products):
        ngram_embeds = x
        products_embeds = self.product_embeds_layer(products)
        products_embeds = products_embeds.unsqueeze(1)
        product_conv = self.product_conv(products_embeds)  # shape: (batch, channel, len)

        product_conv = F.relu(product_conv)

        product_maxpool = F.max_pool1d(product_conv, product_conv.size(2)).squeeze(2)
        ngram_embeds = ngram_embeds.unsqueeze(1)

        conv1 = self.conv1(ngram_embeds)
        conv1 = self.dropout1(conv1)
        max_pool1 = F.max_pool2d(F.relu(conv1), (conv1.shape[2], 1))
        max_pool1 = max_pool1.view(-1, self.num_flat_features(max_pool1))

        conv2 = self.conv2(ngram_embeds)
        conv2 = self.dropout2(conv2)
        max_pool2 = F.max_pool2d(F.relu(conv2), (conv2.shape[2], 1))
        max_pool2 = max_pool2.view(-1, self.num_flat_features(max_pool2))

        conv3 = self.conv3(ngram_embeds)
        conv3 = self.dropout3(conv3)
        max_pool3 = F.max_pool2d(F.relu(conv3), (conv3.shape[2], 1))
        max_pool3 = max_pool3.view(-1, self.num_flat_features(max_pool3))
        flatten = torch.cat((max_pool1, max_pool2, max_pool3, product_maxpool), 1)
        out = self.linear(flatten)
        return out


class Lstm(nn.Module):
    def __init__(self):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=300, hidden_size=600, num_layers=1, batch_first=True)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        output = torch.sum(output, dim=1, keepdim=False)
        return output
