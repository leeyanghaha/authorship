import job_prediction.job_input as jb
from job_prediction.job_input import Input
import torch
import torch.optim as optim
from my_method.my_capsule.models.net import TextCNN


feature_file = '/home/leeyang/research/model/twitter_feature.json'
bert_vocab = '/home/leeyang/research/model/bert/vocab.txt'
tweets = jb.get_user_tweets()
params = {'feature_file': feature_file, 'bert_vocab': bert_vocab, 'feature_dim': 600, 'max_seq_len': 500}

inputs = Input(tweets, batch_size=32, shuffle=True, **params)
model = TextCNN(embedding_dim=300, user_num=inputs.info.num_class)
device = torch.device('cuda:0')
model = torch.nn.DataParallel(model, device_ids=[0, 1])

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00085)



