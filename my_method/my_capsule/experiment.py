import torch
from models.svm_classifier import Svm
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


class Experiment:
    def __init__(self, model, device, criterion, optimizer, epochs):
        self.epochs = epochs
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.use_product = False
        self.only_product = False

    def basic_run(self, train_loader, valid_loader, test_loader):
        self.model.to(self.device)
        for epoch in range(self.epochs):
            correct = 0
            total = 0
            running_loss = 0.0
            print('epoch: {}'.format(epoch + 1), end='')
            for i, review in enumerate(train_loader):
                texts = review['text'].cuda()
                labels = review['user'].cuda()
                if 'product' in review:
                    products = review['product'].cuda()
                total += labels.size(0)
                self.optimizer.zero_grad()
                if self.use_product:
                    output = self.model(texts, products)
                elif self.only_product:
                    output = self.model(products)
                else:
                    output = self.model(texts)
                loss = self.criterion(output, labels)
                _, predicts = torch.max(output.data, 1)
                correct += (labels == predicts).sum().item()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(' loss: %.3f acc: %.3f' %
                  (running_loss / len(train_loader), correct / total), 'correct: {}/{}'.format(correct, total))
            if (epoch + 1) % 3 == 0:
                print('valid: ', end='')
                self.valid(valid_loader)
        print('train finished.')
        print('testing......')
        test_acc = self.valid(test_loader)
        with open('/home/leeyang/results', 'a') as f:
            print('acc %.3f' % test_acc, file=f)

    def valid(self, loader):
        correct = 0
        total = 0
        losses = 0
        self.model.eval()
        with torch.no_grad():
            for i, review in enumerate(loader):
                texts = review['text'].cuda()
                labels = review['user'].cuda()
                if 'product' in review:
                    products = review['product'].cuda()
                if self.use_product:
                    outputs = self.model(texts, products)
                elif self.only_product:
                    outputs = self.model(products)
                else:
                    outputs = self.model(texts)
                _, predict = torch.max(outputs.data, 1)
                total += labels.size(0)
                losses += self.criterion(outputs, labels).item()
                correct += (predict == labels).sum().item()
        print('loss: %.3f acc: %.3f' % (losses / len(loader), (correct / total)),
              'correct: {}/{}'.format(correct, total))
        return correct / total


class BertClassifier(Experiment):
    def __init__(self, inputs, model, device, criterion, optimizer, epochs):
        super(BertClassifier, self).__init__(model, device, criterion, optimizer, epochs)
        self.inputs = inputs

    def run(self):
        train_loader, valid_loader = self.inputs.train_loader, self.inputs.valid_loader
        test_loader = self.inputs.test_loader
        self.basic_run(train_loader, valid_loader, test_loader)


class SvmClassifier:
    def __init__(self, inputs, **param):
        self.classifier = Svm(**param)
        self.info = inputs.info

    def data_split(self):
        train_split = int(self.info.x.shape[0] * 0.6)
        valid_split = train_split + int(self.info.x.shape[0] * 0.2)
        train_x, train_y = self.info.x[: train_split], self.info.users[: train_split]
        valid_x, valid_y = self.info.x[train_split: valid_split], self.info.users[train_split: valid_split]
        test_x, test_y = self.info.x[valid_split:], self.info.users[valid_split:]
        return train_x, train_y, valid_x, valid_y, test_x, test_y

    def run(self):
        train_x, train_y, valid_x, valid_y, test_x, test_y = self.data_split()
        self.classifier.fit(train_x, train_y)
        print('valid: %.3f' % self.valid(valid_x, valid_y))
        total_acc = 0
        for i in range(5):
            acc = self.valid(test_x, test_y)
            total_acc += acc
        print('test: %.3f' % (total_acc / 5))

    def valid(self, x, y):

        y_pred = self.classifier.predict(x)
        acc = self.classifier.accuracy(y, y_pred)
        return acc


class RandomForestsClassifier:
    def __init__(self, inputs, **params):
        self.params = params
        self.rf = RFC(**params['rf_params'])
        self.info = inputs.info

    def data_split(self):
        train_split = int(self.info.x.shape[0] * 0.6)
        valid_split = train_split + int(self.info.x.shape[0] * 0.2)
        train_x, train_y = self.info.x[: train_split], self.info.users[: train_split]
        valid_x, valid_y = self.info.x[train_split: valid_split], self.info.users[train_split: valid_split]
        test_x, test_y = self.info.x[valid_split:], self.info.users[valid_split:]
        return train_x, train_y, valid_x, valid_y, test_x, test_y

    def run(self):
        train_x, train_y, valid_x, valid_y, test_x, test_y = self.data_split()
        grid_params = self.params['grid_params']
        gs_param = {'scoring': 'accuracy', 'param_grid': grid_params, 'estimator': self.rf, 'cv': 5}
        clf = GridSearchCV(**gs_param)
        clf.fit(train_x, train_y)
        print('rf best parameters: ', clf.best_params_)
        print('valid: %.3f' % self.valid(clf, valid_x, valid_y))
        total_acc = 0
        for i in range(5):
            acc = self.valid(clf, test_x, test_y)
            total_acc += acc
        print('test: %.3f' % (total_acc / 5))

    def valid(self, clf, x, y):
        y_pred = clf.predict(x)
        acc = accuracy_score(y_true=y, y_pred=y_pred)
        return acc



