import torch.nn as nn
import utils.function_utils as fu
import collections
import logging
import json
import os
import utils.key_utils as ku
import torch
from torch.utils.data import DataLoader, Dataset
from my_method.my_capsule.models.net import PoolProduct
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
import torch.optim as optim
import numpy as np


# def load_feature_label(split):
#     imi_dir = '/home/leeyang/research/model/imi/'
#     feature_dir = os.path.join(imi_dir, split, 'feature')
#     label_dir = os.path.join(imi_dir, split, 'label')
#     # print('feature_dir', feature_dir)
#     # print('label_dir', label_dir)
#     feature_files = sorted(os.listdir(feature_dir))
#     label_files = sorted(os.listdir(label_dir))
#     features = []
#     labels = []
#     for feature_file, label_file in zip(feature_files, label_files):
#         feature = os.path.join(imi_dir, split, 'feature', feature_file)
#         # print('feature', feature)
#         # print('label', label)
#         label = os.path.join(imi_dir, split, 'label', label_file)
#         # if split != 'train':
#         #     print(torch.load(feature).cpu().detach().numpy().shape)
#         features.append(torch.load(feature).cpu().detach().numpy())
#         labels.append(torch.load(label).cpu().detach().numpy())
#     return features, labels

def get_feature(split):
    product_feature_dir = os.path.join('/home/leeyang/research/model/imi/feature_for_product', split)
    user_feature_dir = os.path.join('/home/leeyang/research/model/imi/feature_for_user', split)
    product_feature_files = sorted(os.listdir(product_feature_dir))
    user_feature_files = sorted(os.listdir(user_feature_dir))
    features = []
    for product_feature_file, user_feature_file in zip(product_feature_files, user_feature_files):
        product_feature_path = os.path.join(product_feature_dir, product_feature_file)
        user_feature_path = os.path.join(user_feature_dir, user_feature_file)
        product_feature = torch.load(product_feature_path).cpu().detach().numpy()
        user_feature = torch.load(user_feature_path).cpu().detach().numpy()
        # features.append(user_feature + product_feature)
        features.append(np.concatenate((product_feature, user_feature), axis=1))
    return features


def get_labels(split):
    user_label_dir = os.path.join('/home/leeyang/research/model/imi/user_label', split)
    user_label_files = sorted(os.listdir(user_label_dir))
    labels = []
    for user_label_file in user_label_files:
        user_label_path = os.path.join(user_label_dir, user_label_file)
        user_label = torch.load(user_label_path).cpu().detach().numpy()
        labels.append(user_label)
    return labels


# features = get_feature('test')

# print('len: ', len(features))
# print('feature0', features[0])
# print('size', features[0].shape)


# labels = get_labels('test')

# print('label len: ', len(labels))
# print('label0', labels[0])
# print('size', labels[0].shape)


def load_feature_label(split):
    features = get_feature(split)
    labels = get_labels(split)
    return features, labels


def stack_feature_label(features, labels):
    for i, feature in enumerate(features):
        if feature.shape[0] != 32:
            del features[i]
            del labels[i]
    stacked_feature = np.stack(np.array(features), axis=0).reshape((-1, 900*2))
    stacked_label = np.stack(np.array(labels), axis=0).reshape(-1)
    return stacked_feature, stacked_label


train_features, train_labels = load_feature_label('train')
train_features, train_labels = stack_feature_label(train_features, train_labels)
valid_features, valid_labels = load_feature_label('valid')
# for feature in valid_features:
#     print(feature.shape)
valid_features, valid_labels = stack_feature_label(valid_features, valid_labels)
test_features, test_labels = load_feature_label('test')
test_features, test_labels = stack_feature_label(test_features, test_labels)

print('feature0', train_features.shape)
print('label0', train_labels.shape)


class FeatureDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        text = self.x[idx]
        user = self.y[idx]
        return {'text': text, 'user': user}


train_dataset = FeatureDataset(train_features, train_labels)
valid_dataset = FeatureDataset(valid_features, valid_labels)
test_dataset = FeatureDataset(test_features, test_labels)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=5)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=5)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=5)

model = PoolProduct(161)
device = torch.device('cuda:0')
# model = torch.nn.DataParallel(model, device_ids=[0, 1])
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


def train(epochs=50):
    model.to(device)
    for epoch in range(epochs):
        correct = 0
        total = 0
        running_loss = 0.0
        print('epoch: {}'.format(epoch+1), end='')
        for i, review in enumerate(train_loader):
            texts = review['text'].cuda()
            labels = review['user'].cuda()
            # users = review['user'].cuda()
            # torch.save(labels, '/home/leeyang/research/model/imi/train/label/{}.pt'.format(i))
            total += labels.size(0)
            optimizer.zero_grad()
            output = model(texts)
            loss = criterion(output, labels)
            _, predicts = torch.max(output.data, 1)
            # print(labels)
            correct += (labels == predicts).sum().item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(' loss: %.3f acc: %.3f' %
              (running_loss / len(train_loader), correct / total), 'correct: {}/{}'.format(correct, total))
        if (epoch + 1) % 3 == 0:
            print('valid: ', end='')
            valid(model, 'valid')
    print('train finished.')
    print('testing......')
    valid(model, 'test')
    return model


def valid(model, split):
    correct = 0
    total = 0
    loader = valid_loader if split == 'valid' else test_loader
    losses = 0
    # model.eval()
    with torch.no_grad():
        for i, review in enumerate(loader):
            texts, labels = review['text'].cuda(), review['user'].cuda()
            # torch.save(labels, '/home/leeyang/research/model/imi/{}/label/{}.pt'.format(split, i))
            outputs = model(texts)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            # print('label: ', labels)
            # print('predict: ', predict)
            losses += criterion(outputs, labels).item()
            correct += (predict == labels).sum().item()
    print('loss: %.3f acc: %.3f' % (losses / len(test_loader), (correct / total)),
          'correct: {}/{}'.format(correct, total))


if __name__ == '__main__':
    train(50)
