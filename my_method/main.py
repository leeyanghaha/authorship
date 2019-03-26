from my_method.input.review_input import Input, ReviewDataSet
import utils.function_utils as fu
import utils.key_utils as ku
from torch.utils.data import DataLoader
from my_method.my_capsule.models.net import BertCNN, TextCNN, BertProduct
import torch
import torch.optim as optim
from my_method.my_capsule.experiment import Experiment, SvmClassifier, RandomForestsClassifier
from sklearn.utils import shuffle






class ExperimentWrapper:
    def __init__(self, method):
        self.method = method
        self.device = torch.device('cuda:0')
        self.reviews = self.get_reviews()
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_reviews(self):
        file = '/home/leeyang/research/data/Movie.json'
        return shuffle(fu.load_array(file))

    def run(self):
        use_cuda = False
        if self.method == ku.svm:
            svm_params = {'penalty': 'l2', 'loss': 'hinge', 'dual': True, 'multi_class': 'ovr',
                          'fit_intercept': True, 'max_iter': 10000}
            inputs = Input(self.reviews, method=self.method)
            experiment = SvmClassifier(inputs, **svm_params)
            experiment.run()

        elif self.method == ku.cnn:
            use_cuda = True
            params = {'min_threshold': 6, 'max_seq_len': 3500, 'feature_name': 'n-gram'}
            inputs = Input(self.reviews, method=self.method, pretrained=False, batch_size=32, shuffle=True, **params)
            model = TextCNN(inputs.info.vocab_size, embedding_dim=300, user_num=inputs.info.user_num)

        elif self.method == ku.bert:
            use_cuda = True
            bert_vocab = '/home/leeyang/research/model/bert/vocab.txt'
            feature_file = '/home/leeyang/research/model/feature_last_300.json'
            params = {'feature_file': feature_file, 'feature_dim': 300, 'max_seq_len': 1500, 'bert_vocab': bert_vocab}
            inputs = Input(self.reviews, method=ku.bert, pretrained=True, batch_size=32, shuffle=True, **params)
            model = BertCNN(embedding_dim=300, user_num=inputs.info.user_num)

        elif self.method == ku.bert_product:
            use_cuda = True
            bert_vocab = '/home/leeyang/research/model/bert/vocab.txt'
            feature_file = '/home/leeyang/research/model/feature_last_300.json'
            params = {'feature_file': feature_file, 'feature_dim': 300, 'max_seq_len': 1500, 'bert_vocab': bert_vocab}
            inputs = Input(self.reviews, method=ku.bert, pretrained=True, batch_size=32, shuffle=True, **params)
            model = BertProduct(embedding_dim=300, user_num=inputs.info.user_num, product_num=inputs.info.product_num)

        elif self.method == ku.rf:
            grid_params = {'n_estimators': [5000], 'criterion': ['gini'],
                           'max_depth': [1000], 'min_samples_split': [3],
                           'min_samples_leaf': [3], 'max_features': ['sqrt', 'log2'], 'max_leaf_nodes': [None],
                           'min_impurity_decrease': [0.], 'random_state': [1]}
            rf_params = {'n_jobs': 10, 'verbose': 0, 'warm_start': False}
            params = {'grid_params': grid_params, 'rf_params': rf_params}
            inputs = Input(self.reviews, method=self.method)
            experiment = RandomForestsClassifier(inputs, **params)
            experiment.run()

        elif self.method == ku.lda:
            pass

        else:
            raise NotImplementedError('{} not implemented'.format(self.method))

        if use_cuda:
            model = torch.nn.DataParallel(model, device_ids=[0, 1])
            optimizer = optim.Adam(model.parameters(), lr=0.00085)
            experiment = Experiment(model, self.device, self.criterion, optimizer, epochs=70)
            if self.method == ku.bert_product:
                experiment.use_product = True
            train_loader, valid_loader, test_loader = inputs.train_loader, inputs.valid_loader, inputs.test_loader
            experiment.basic_run(train_loader, valid_loader, test_loader)



# bert_vocab = '/home/leeyang/research/model/bert/vocab.txt'
# feature_file = '/home/leeyang/research/model/feature_last_300.json'
#
# pretrained_params = {'feature_file': feature_file, 'feature_dim': 300, 'max_seq_len': 1500, 'bert_vocab': bert_vocab}
# nonpretrained_params = {'min_threshold': 6, 'max_seq_len': 3500, 'feature_name': 'n-gram'}
#
# # inputs = Input(reviews, pretrained=True, batch_size=32, shuffle=True, **pretrained_params)
# inputs = Input(reviews, pretrained=False, batch_size=32, shuffle=True, **nonpretrained_params)
#
# model = TextCNN(vocab_size=inputs.info.vocab_size, embedding_dim=300, user_num=inputs.info.user_num)
#
# # model = TextCNN(embedding_dim=300, user_num=inputs.info.user_num)
# # model = TextProductCNN(embedding_dim=300, user_num=inputs.info.user_num, product_num=inputs.info.product_num)
# device = torch.device('cuda:0')
# model = torch.nn.DataParallel(model, device_ids=[0, 1])
#
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.00085)
#
# experiment = Experiment(model=model, device=device, criterion=criterion, optimizer=optimizer, epochs=70)


if __name__ == '__main__':
    # train_loader, valid_loader, test_loader = inputs.train_loader, inputs.valid_loader, inputs.test_loader
    # experiment.run_experiment(train_loader, valid_loader, test_loader)
    exp = ExperimentWrapper(method=ku.rf)
    exp.run()


