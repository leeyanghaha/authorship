from baselines.trying.input import ReviewDataSet
from baselines.trying.net import TextCNN
from utils.data_utils import ReviewLoader
import utils.key_utils as ku
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import os
import torch
from utils.data_utils import ReviewLoader, UserHelper, DataHelper
import utils.function_utils as fu


userhelper = UserHelper()
datahelper = DataHelper()
feature = 'n-gram'

# review_loader = ReviewLoader(ku.Movie, product_num=100)
# reviews = review_loader.get_data()


def get_reviews():
    file = r'/home/leeyang/research/data/Movie.json'
    reviews = fu.load_array(file)
    return reviews


reviews = get_reviews()
train_dataset = ReviewDataSet(reviews, 'train', feature=feature, max_len=500)
user2idx = train_dataset.user2idx
user_num = len(user2idx)
vocab_size = train_dataset.vocab_size
max_len = train_dataset.max_len

# products = datahelper.get_products(reviews)
# product2idx = datahelper.product2idx(products)
# products_id = review_loader.load_products_id(products, product2idx)


embedding_dim = 300


model = TextCNN(max_len=max_len, embedding_dim=embedding_dim, vocab_size=vocab_size,
                     user_num=user_num)
device_ids = [0, 1]
device = torch.device('cuda:0')
model = nn.DataParallel(model, device_ids=device_ids)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('total_params: ', total_params)
# review_loader = DataLoader(review_dataset, batch_size=32, shuffle=True, num_workers=4)


def train(epochs=50):

    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, review in enumerate(train_loader):
            texts = review['text'].cuda()
            labels = review['label'].cuda()
            total += labels.size(0)
            optimizer.zero_grad()
            outputs = model(texts)
            _, predicts = torch.max(outputs.data, 1)
            correct += (labels == predicts).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('-' * 6)
        print('epoch %d: loss: %.3f acc: %.3f' %
              (epoch + 1, running_loss / len(train_loader), correct / total))
        if (epoch + 1) % 3 == 0:
            valid(model, 'valid')
    print('train finished.')
    print('testing......')
    valid(model, 'test')
    return model


def valid(model, split):
    correct = 0
    total = 0
    test_dataset = ReviewDataSet(reviews, split)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    losses = 0
    model.eval()
    with torch.no_grad():
        for i, review in enumerate(test_loader):
            texts, labels = review['text'].cuda(), review['label'].cuda()
            outputs = model(texts)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            losses += criterion(outputs, labels)
            correct += (predict == labels).sum().item()
    print('acc: %.3f loss: %.3f' % ((correct / total), losses / len(test_loader)))


if __name__ == '__main__':
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    train()
