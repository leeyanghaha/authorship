import torch
import torch.nn as nn
import torch.optim as optim
import my_method.my_capsule.models.layers as layers
import torch.nn.functional as F
from baselines.trying.net import TextCNN


class CapsuleModel(nn.Module):
    def __init__(self, vocab_size, num_classes, hyparam):
        super(CapsuleModel, self).__init__()
        self.hyparam = hyparam
        self.vocabs_size = vocab_size
        self.num_classes = num_classes
        self.init_conv = nn.Conv2d(1, 256, (3, self.hyparam.embedding_dim))
        self.embedding_layer = nn.Embedding(self.vocabs_size, self.hyparam.embedding_dim)
        self.capsule_conv = layers.CapsuleConv(input_dim=256, out_dim=10, out_atoms=8)
        self.clf_layer = layers.ClfCapsule(self.num_classes, 8, 16)
        self.softmax = nn.Softmax(dim=1)
        self.relu = self.activations('relu')

    @staticmethod
    def activations(active):
        if active == 'relu':
            return nn.ReLU()

    @staticmethod
    def reshape_embedding(input):
        batch, height, width = input.size()
        reshaped = input.view(batch, 1, height, width)
        return reshaped

    @staticmethod
    def extend_dim(input):
        batch, channel, heigth, width = input.size()
        extended = input.view(batch, 1, channel, heigth, width)
        return extended

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i
        return num_features

    def forward(self, input):
        text_embedding = self.embedding_layer(input)
        reshaped_embedding = self.reshape_embedding(text_embedding)
        conv1 = self.init_conv(reshaped_embedding)

        # conv1 = self.extend_dim(self.relu(conv1))
        # # shape: (batch, out_dim, out_atoms, height, width) = (64, 8, 8, 1748, 1)
        #
        capsule_conv = self.capsule_conv(conv1)
        # batch, out_atoms, out_dim, height, width = u.size()
        # capsule_conv = capsule_conv.view(batch, out_dim, out_atoms, height, width)
        # print('capsule_conv', capsule_conv.size())

        # print('capsule_conv', capsule_conv.size())
        # # conv1_flatten = conv1_capule1.view(-1, self.num_flat_features(conv1_capule1))
        out_activation = self.clf_layer(capsule_conv)
        # # print(conv1_flatten.size())
        # print('out_activation', out_activation.size())
        out = torch.sqrt((out_activation ** 2).sum(dim=2, keepdim=True)).squeeze(2).squeeze(2)
        # out = torch.norm(out_activation, dim=2)
        # out = self.liner(conv1_flatten)
        # print('out', out.size())
        return out


class Test(TextCNN):
    def __init__(self, vocab_size, num_classes):
        super(Test, self).__init__(500, 300, vocab_size, num_classes)




