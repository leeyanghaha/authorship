"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829
PyTorch implementation by Kenta Iwasaki @ Gram.AI.
"""
import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.autograd import Variable


BATCH_SIZE = 100
NUM_CLASSES = 10
NUM_EPOCHS = 500
NUM_ROUTING_ITERATIONS = 3


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))


        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])
            # self.capsules = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            print('x.size: ', x.size())
            print('route_weights.size: ', self.route_weights.size())
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size())).cuda()
            # print('logits.size: ', logits.size())
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            print('params: ', sum(p.numel() for p in self.capsules.parameters() if p.requires_grad))
            # print('capsule x.size()', capsule(x).size())
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            # outputs = [capsule(x) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)
            # print()
        return outputs


class CapsuleNet(nn.Module):
    def __init__(self, vocab_size, num_classes, hyparam):
        super(CapsuleNet, self).__init__()
        self.vocab_size = vocab_size
        self.hyparam = hyparam
        self.num_classes = num_classes
        self.embedding_layer = nn.Embedding(self.vocab_size, self.hyparam.embedding_dim)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(3, self.hyparam.embedding_dim), stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=2, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=(3, 1), stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=self.num_classes, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)

        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    @staticmethod
    def reshape_embedding(input):
        batch, height, width = input.size()
        reshaped = input.view(batch, 1, height, width)
        return reshaped

    def forward(self, x, y=None):
        x = self.embedding_layer(x)
        x = self.reshape_embedding(x)
        x = F.relu(self.conv1(x), inplace=True)
        print('conv1.size: ', x.size())
        x = self.primary_capsules(x)
        print('primary_capsules.size: ', x.size())
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(self.num_classes)).cuda().index_select(dim=0, index=max_length_indices.data)

        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        return classes, reconstructions

