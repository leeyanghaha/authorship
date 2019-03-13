import torch
class Experiment:
    def __init__(self, input, model, device, criterion, optimizer, epochs):
        self.epochs = epochs
        self.model = model
        self.input = input
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self):
        self.model.to(self.device)
        for epoch in range(self.epochs):
            correct = 0
            total = 0
            running_loss = 0.0
            print('epoch: {}'.format(epoch + 1), end='')
            for i, review in enumerate(self.input.train_loader):
                texts = review['text'].cuda()
                labels = review['user'].cuda()
                products = review['product'].cuda()
                # torch.save(products, '/home/leeyang/research/model/imi/product_label/train/{}.pt'.format(i))
                total += labels.size(0)
                self.optimizer.zero_grad()
                output = self.model(texts)
                loss = self.criterion(output, labels)
                _, predicts = torch.max(output.data, 1)
                correct += (labels == predicts).sum().item()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(' loss: %.3f acc: %.3f' %
                  (running_loss / len(self.input.train_loader), correct / total), 'correct: {}/{}'.format(correct, total))
            if (epoch + 1) % 3 == 0:
                print('valid: ', end='')
                self.valid('valid')
        print('train finished.')
        print('testing......')
        self.valid('test')

    def valid(self, split):
        correct = 0
        total = 0
        loader = self.input.valid_loader if split == 'valid' else self.input.test_loader
        losses = 0
        test_times = 5 if split == 'test' else 1
        self.model.eval()
        with torch.no_grad():
            for _ in range(test_times):
                for i, review in enumerate(loader):
                    texts = review['text'].cuda()
                    labels = review['user'].cuda()
                    products = review['product'].cuda()
                    # torch.save(products, '/home/leeyang/research/model/imi/product_label/{}/{}.pt'.format(split, i))
                    outputs = self.model(texts)
                    _, predict = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    # print('label: ', labels)
                    # print('predict: ', predict)
                    losses += self.criterion(outputs, labels).item()
                    correct += (predict == labels).sum().item()
        print('loss: %.3f acc: %.3f' % (losses / len(self.input.test_loader) / test_times, (correct / total)),
              'correct: {}/{}'.format(int(correct / test_times), int(total / test_times)))
