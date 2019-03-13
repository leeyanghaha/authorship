from my_method.input.review_input import Input, ReviewDataSet
import utils.function_utils as fu
from torch.utils.data import DataLoader
from my_method.my_capsule.models.net import TextCNN
import torch
import torch.optim as optim
from my_method.my_capsule.experiment import Experiment


def get_reviews():
    file = '/home/leeyang/research/data/Movie.json'
    return fu.load_array(file)


reviews = get_reviews()
bert_vocab = '/home/leeyang/research/model/bert/vocab.txt'
feature_file = '/home/leeyang/research/model/feature_last_300.json'

pretrained_params = {'feature_file': feature_file, 'feature_dim': 300, 'max_seq_len': 1500, 'bert_vocab': bert_vocab}
nonpretrained_params = {'min_threshold': 6, 'max_seq_len': 3500, 'feature_name': 'n-gram'}

inputs = Input(reviews, pretrained=True, batch_size=32, shuffle=True, **pretrained_params)


model = TextCNN(embedding_dim=300, user_num=inputs.info.user_num)
device = torch.device('cuda:0')
model = torch.nn.DataParallel(model, device_ids=[0, 1])
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


experiment = Experiment(input=inputs, model=model, device=device, criterion=criterion, optimizer=optimizer, epochs=35)


if __name__ == '__main__':
    experiment.train()


