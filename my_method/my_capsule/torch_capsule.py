from utils.time_utils import Time
from my_method.my_capsule.input import ReviewDataSet, ReviewInfo
from utils.data_utils import ReviewLoader
import utils.key_utils as ku
from torch.utils.data import DataLoader
from my_method.my_capsule.models.net import CapsuleModel as cpm
import torch
import torch.optim as optim
import torch.nn as nn
import utils.function_utils as fu


def get_reviews():
    file = r'/home/leeyang/research/data/Movie.json'
    reviews = fu.load_array(file)
    return reviews


def train(epochs=50):
    model.to(device)
    for epoch in range(epochs):
        correct = 0
        total = 0
        running_loss = 0.0
        print('epoch: {}'.format(epoch+1), end='')
        for i, review in enumerate(train_loader):
            texts = review['text'].cuda()
            labels = review['label'].cuda()
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
              (running_loss / len(train_loader), correct / total))
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
            texts, labels = review['text'].cuda(), review['label'].cuda()
            outputs = model(texts)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            losses += criterion(outputs, labels).item()
            correct += (predict == labels).sum().item()
    print('loss: %.3f acc: %.3f' % (losses / len(test_loader), (correct / total)))


if __name__ == '__main__':
    reviews = get_reviews()
    # reviews = ReviewLoader(ku.Movie, product_num=50).get_data()
    review_info = ReviewInfo(reviews, max_len=500)

    num_classes = review_info.num_classes
    vocab_size = review_info.vocab_size

    train_set = ReviewDataSet(review_info.x, review_info.y, 'train')
    valid_set = ReviewDataSet(review_info.x, review_info.y, 'valid')
    test_set = ReviewDataSet(review_info.x, review_info.y, 'test')

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=5)
    valid_loader = DataLoader(valid_set, batch_size=32, shuffle=True, num_workers=5)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=5)

    device_ids = [0, 1]
    device = torch.device('cuda:0')
    # model = Test(vocab_size, num_classes)
    model = cpm(vocab_size, num_classes, hyparam)
    # model = CapsuleNet(vocab_size, num_classes, hyparam)
    model = nn.DataParallel(model, device_ids=device_ids)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # for name, param in model.named_parameters():
    #     print(name)
    print('total_params: ', total_params)

    train()




