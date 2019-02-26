import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def squash(s, layer):
    # This is equation 1 from the paper.
    if layer == 'conv':
        dim = 1
    else:
        dim = 2
    mag_sq = torch.sum(s ** 2, dim=dim, keepdim=True)
    mag = torch.sqrt(mag_sq)
    s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
    return s



def _leaky_routing(logits, output_dim, softmax_layer):
    leak = torch.zeros_like(logits).cuda()
    leak = torch.sum(leak, dim=2, keepdim=True)
    leak_logits = torch.cat((leak, logits), dim=2)
    leaky_routing = softmax_layer(leak_logits)
    return torch.split(leaky_routing, [1, output_dim], dim=2)[1]


class CapsuleConv(nn.Module):
    def __init__(self,
                 input_dim,
                 out_dim,
                 input_atoms=8,
                 out_atoms=8,
                 stride=2,
                 kernel_size=5,
                 **routing):
        super(CapsuleConv, self).__init__()
        self.input_dim = input_dim
        self.input_atoms = input_atoms
        self.out_dim = out_dim
        self.out_atoms = out_atoms
        self.conv_capsule1 = nn.Conv2d(input_dim, out_atoms * out_dim, (kernel_size, 1), stride=stride)

    def forward(self, input_tensor):
        input_shape = input_tensor.size()
        batch, _, in_height, in_width = input_shape
        # input_tensor = input_tensor.view(self.input_dim * batch, self.input_atoms, in_height, in_width)
        conv = self.conv_capsule1(input_tensor)
        conv_shape = conv.size()
        # print('conv.shape', conv_shape)
        _, _, conv_height, conv_width = conv_shape
        conv_reshaped = conv.view(batch, self.out_atoms, conv_height * conv_width * self.out_dim)
        return squash(conv_reshaped, layer='conv')


class ConvUnit(nn.Module):
    def __init__(self,
                 input_dim,
                 out_dim,
                 kernel_size,
                 stride,
                 ):
        super(ConvUnit, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=input_dim,
                               out_channels=out_dim,
                               kernel_size=(kernel_size, 1),
                               stride=stride,
                               bias=True)

    def forward(self, input):
        return self.conv0(input)


# class CapsuleConv(nn.Module):
#     def __init__(self,
#                  input_dim,
#                  out_atoms,
#                  out_dim,
#                  kernel_size=5,
#                  stride=2,
#                  ):
#         super(CapsuleConv, self).__init__()
#         self.out_atoms = out_atoms
#         def create_conv_unit(unit_idx):
#             unit = ConvUnit(input_dim=input_dim,
#                             out_dim=out_dim,
#                             kernel_size=kernel_size,
#                             stride=stride)
#             self.add_module("unit_" + str(unit_idx), unit)
#             return unit
#         self.units = [create_conv_unit(i) for i in range(out_atoms)]
#         # self.units = [ConvUnit(input_dim=input_dim,
#         #                        out_dim=out_dim,
#         #                        kernel_size=kernel_size,
#         #                        stride=stride) for _ in range(out_atoms)]
#         self.conv_unit = ConvUnit(input_dim=input_dim,
#                             out_dim=out_dim,
#                             kernel_size=kernel_size,
#                             stride=stride)
#
#     def forward(self, input):
#         u = []
#         for i in range(self.out_atoms):
#             unit_i = self.conv_unit(input)
#             u.append(unit_i)
#         u = [self.units[i](input) for i in range(self.out_atoms)]
#         # print('u_0', u[0].size())
#         u = torch.stack(u, dim=1)
#
#         # return squash(u).view(batch, out_dim, out_atoms, height, width)
#         return squash(u.view(input.size(0), self.out_atoms, -1))

class ClfCapsule(nn.Module):
    def __init__(self, num_classes, input_atoms, out_atoms):
        super(ClfCapsule, self).__init__()
        self.num_classes = num_classes
        self.out_atoms = out_atoms
        self.input_atoms = input_atoms
        self.W = nn.Parameter(torch.randn(1, 1, self.num_classes, out_atoms, self.input_atoms))
        self.softmax = nn.Softmax(dim=2)

    def dynamic_routing(self,
                        input_tensor,
                        mul,
                        num_classes,
                        num_routing):
        b_ij = Variable(torch.zeros((1, mul, num_classes, 1))).cuda()
        # self.W = nn.Parameter(torch.randn(1, mul, self.num_classes, out_atoms, self.input_atoms)).cuda()
        batch_size = input_tensor.size(0)
        # print(mul)
        input_tensor = torch.stack([input_tensor] * self.num_classes, dim=2).unsqueeze(4)
        # print('input', input_tensor.size())
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, input_tensor)
        for i in range(num_routing):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            v_j = squash(s_j, layer='fc')
            v_j1 = torch.cat([v_j] * mul, dim=1)

            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            b_ij = b_ij + u_vj1
        return v_j.squeeze(1)

    def forward(self, input):
        # print(input.size())
        batch, input_atoms, mul = input.size()
        input_tensor = input.view(batch, mul, input_atoms)
        # print('input_tensor', input_tensor.size())
        activation = self.dynamic_routing(input_tensor=input_tensor,
                                          mul=mul,
                                          num_classes=self.num_classes,
                                          num_routing=3)
        # print('activation.size: ', activation.size())
        # activation = activation.view(batch, self.num_classes, self.out_atoms)
        return activation



