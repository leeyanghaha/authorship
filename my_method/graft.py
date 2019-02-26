import torch
import torch.nn as nn
from torch.autograd import Variable
out_atoms = 4
input_atoms = 8
batch = 16
num_classes = 9
mul = 12

weight = torch.rand((num_classes, input_atoms, out_atoms))
input = torch.rand((batch * mul, input_atoms))

# # print(weight.size())
# # print(input.size())
u_hat = torch.matmul(input, weight)

u_hat = u_hat.view(batch, mul, num_classes, out_atoms)
# print(u_hat.size())
c_ij = torch.rand((batch, mul, num_classes, 1))
c_ij_u_hat = torch.mul(c_ij, u_hat)
s = torch.sum(c_ij_u_hat, dim=1)

s = s.unsqueeze(2)
# print(s.size())
# print(torch.mul(s, u_hat).size())

# print(c_ij.size())
# print((c_ij.mul(u_hat)).size())
#
# s = torch.sum(c_ij.mul(u_hat), dim=1)
# u_hat = u_hat.view(mul * batch, num_classes,  out_atoms, 1)
# s = s.unsqueeze(1)
# print(s.size())
# print(u_hat.size())
# print(torch.matmul(s, u_hat).size())
# v = torch.sum(u_hat, dim=1)
# print(v.size())


def dynamic_routing(x):
    b_ij = Variable(torch.rand((batch, mul, num_classes)))


    for i in range(3):
        c_i = nn.Softmax(dim=1)(b_ij).unsqueeze(3)
        u_hat = torch.matmul(input, weight)
        u_hat = u_hat.view(batch, mul, num_classes, out_atoms)
        print('u_hat', u_hat.size())
        # c_ij = torch.rand((batch, mul, num_classes, 1))
        c_ij_u_hat = torch.mul(c_i, u_hat)
        s_j = torch.sum(c_ij_u_hat, dim=1, keepdim=True).unsqueeze(4)
        # print(s_j.size())
        v_j1 = torch.cat([s_j] * mul, dim=1)
        u_hat = u_hat.unsqueeze(4)
        # print('v_j1', v_j1.size())
        u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).squeeze(3)
        # print(u_vj1.size())
        b_ij = b_ij + u_vj1
# v_j1 = torch.rand(batch, mul, num_classes, out_atoms, 1)
# u_hat = torch.rand(batch, mul, num_classes, 1, out_atoms)
#
# # u_vj1 = torch.matmul(u_hat, v_j1).squeeze(4).mean(dim=0, keepdim=True)
# u_vj1 = torch.matmul(u_hat, v_j1).squeeze(4).squeeze(3)
# print(u_vj1.size())

if __name__ == '__main__':
    dynamic_routing(input)
