import torch.nn as nn
import torch
# import math
# import os
# import numpy as np
from einops import rearrange
# import copy
# from sklearn.metrics import mutual_info_score
# from torch import Tensor
#
# from torch_geometric.nn.inits import glorot, zeros
# from torch_geometric.typing import (Adj, Size, OptTensor, PairTensor)
# from typing import Union, Tuple, Optional
# from torch_sparse import SparseTensor, set_diag
# from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights


# 1 将长时序的脑电信号进行压缩
# class CompressNet(nn.Module):
#     def __init__(self):
#         super(CompressNet, self).__init__()
#
#         # [64,1,32,512] -> [64,4,32,64]
#         self.conv_1_1 = nn.Sequential(nn.Conv2d(in_channels=1,
#                                                 out_channels=4,
#                                                 kernel_size=(1, 64),
#                                                 stride=1),
#                                       # nn.BatchNorm1d(num_features=4),
#                                       nn.AvgPool2d(kernel_size=(1, 9), stride=(1, 9)),
#                                       nn.LeakyReLU(0.15))
#
#         # [64,4,32,64] -> [64,8,32,8]
#         self.conv_1_2 = nn.Sequential(nn.Conv2d(in_channels=4,
#                                                 out_channels=8,
#                                                 kernel_size=(1, 32),
#                                                 stride=1),
#                                       # nn.BatchNorm1d(num_features=8),
#                                       nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 3)),
#                                       nn.LeakyReLU(0.15))
#
#         # [64,1,32,512] -> [64,4,32,8]
#         self.conv_2_1 = nn.Sequential(nn.Conv2d(in_channels=1,
#                                                 out_channels=4,
#                                                 kernel_size=(1, 128),
#                                                 stride=(1, 24)),
#                                       # nn.BatchNorm1d(num_features=4),
#                                       nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
#                                       nn.LeakyReLU(0.15))
#         self.bn_2D = nn.BatchNorm2d(num_features=12)
#
#         # self.mlp = nn.Linear(3072, 512)
#
#     def forward(self, x):
#         out_1_1 = self.conv_1_1(x)
#         out_1_2 = self.conv_1_2(out_1_1)
#         out_2_1 = self.conv_2_1(x)
#         out = torch.cat((out_1_2, out_2_1), dim=1)
#         out = self.bn_2D(out)
#         # out = out.view(out.size(0), -1)
#         # out = self.mlp(out)
#         # 对合成的特征进行批归一化，降低各个部分之间的差异
#
#         return out

class CompressNet(nn.Module):
    def __init__(self):
        super(CompressNet, self).__init__()

        # [128,1,22,438] -> [128,4,22,44]
        self.conv_1_1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                                out_channels=4,
                                                kernel_size=(1, 32),
                                                stride=1),
                                      # nn.BatchNorm1d(num_features=4),
                                      nn.AvgPool2d(kernel_size=(1, 10), stride=(1, 10)),
                                      nn.LeakyReLU(0.15))

        # [128,4,22,44] -> [128,8,22,8]
        self.conv_1_2 = nn.Sequential(nn.Conv2d(in_channels=4,
                                                out_channels=8,
                                                kernel_size=(1, 16),
                                                stride=1),
                                      # nn.BatchNorm1d(num_features=8),
                                      nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 3)),
                                      nn.LeakyReLU(0.15))

        # [128,1,22,438] -> [128,4,22,8]
        self.conv_2_1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                                out_channels=4,
                                                kernel_size=(1, 64),
                                                stride=(1, 12)),
                                      # nn.BatchNorm1d(num_features=4),
                                      nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
                                      nn.LeakyReLU(0.15))
        self.bn_2D = nn.BatchNorm2d(num_features=12)
    def forward(self, x):
        out_1_1 = self.conv_1_1(x)
        out_1_2 = self.conv_1_2(out_1_1)
        out_2_1 = self.conv_2_1(x)
        out = torch.cat((out_1_2, out_2_1), dim=1)
        out = self.bn_2D(out)
        # out = out.view(out.size(0), -1)
        # out = self.mlp(out)
        # 对合成的特征进行批归一化，降低各个部分之间的差异

        return out
# 2 运行位置相关的自注意力
class RelationAwareness(nn.Module):
    def __init__(self):
        super(RelationAwareness, self).__init__()

        self.expand_size = 8
        self.a = nn.Parameter(torch.empty(size=(2 * self.expand_size, 1)))##作为nn.Module中的可训练参数使用。它与torch.Tensor的区别就是nn.Parameter会自动被认为是module的可训练参数，即加入到parameter()这个迭代器中去
        nn.init.xavier_uniform_(self.a.data, gain=1.414)#用于初始化神经网络的权重。

        self.bn_2D = nn.BatchNorm2d(12)


    def forward(self, feature):
        # location_embed = self.location_em(location) # 可能考虑对location进行适当的归一化
        # feature_local_embed = self.relu(feature_embed + location_embed) # 包含位置编码的输入

        h1 = torch.matmul(feature, self.a[:self.expand_size, :])
        h2 = torch.matmul(feature, self.a[self.expand_size:, :])
        ##填补少的维度，对齐-3维度，-1和-2维度作矩阵乘法
        # broadcast add
        h2_T = rearrange(h2, "b n h d -> b n d h")
        e = h1 + h2_T
        e = self.bn_2D(e)

        # e_mean1 = e.mean(dim=1, keepdim=True)
        # e1 = e_mean1
        # # print(e1)
        # e2 = torch.squeeze(e1)
        # print(e2.size())
        # e2 = e2.cpu().detach().numpy()
        # np.save(os.path.join("/data2/zdn/deap_chord/", (str(3) + ".npy")), e2)
        # print(np.load(os.path.join("/data2/zdn/deap_chord/", (str(1) + ".npy"))))
        return e


# 3 使用深度2D backbone进行特征提取
class Res18(nn.Module):
    def __init__(self):
        super(Res18, self).__init__()
        self.resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT  old(pretrained=True)

        self.resnet18.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3,
                                        bias=False)
        self.resnet18.fc = nn.Linear(512, 512)

    def forward(self, x):
        x = self.resnet18(x)
        return x


class FcNet(nn.Module):
    def __init__(self, class_num):
        super(FcNet, self).__init__()

        self.dropout = nn.Dropout(0.8)
        self.l_relu = nn.LeakyReLU(0.1)
        self.bn = nn.BatchNorm1d(128)
        self.mlp_0 = nn.Linear(512, 128)  #(512,128)
        self.mlp_1 = nn.Linear(128, class_num)

    def forward(self, x):
        x = self.dropout(x)
        x = self.mlp_0(x)

        x = self.l_relu(x)
        # x = self.bn(x)
        x = self.mlp_1(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, emb_size, args, cifar_flag=False):
        super(ConvNet, self).__init__()
        # set size
        self.hidden = 128
        self.last_hidden = self.hidden * 1 if not cifar_flag else self.hidden
        # self.last_hidden = self.hidden * 1 if not cifar_flag else self.hidden
        self.emb_size = emb_size
        self.args = args


        # set layers#32→16
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=12,
                                              out_channels=self.hidden,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        #16→7
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                              out_channels=int(self.hidden*1.5),
                                              kernel_size=3,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden*1.5)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        #7→3
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden*1.5),
                                              out_channels=self.hidden*2,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.4))
        self.max = nn.MaxPool2d(kernel_size=2)

        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden*2,
                                              out_channels=self.hidden*4,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))
        self.layer_second = nn.Sequential(nn.Linear(in_features=self.last_hidden * 2,   #in_features (int) – size of each input sample
                                                    out_features=self.emb_size, bias=True),
                                          nn.BatchNorm1d(self.emb_size))
        self.layer_last = nn.Sequential(nn.Linear(in_features=self.last_hidden * 4,
                                                  out_features=self.emb_size, bias=True),
                                        nn.BatchNorm1d(self.emb_size))

    def forward(self, input_data):
        out_1 = self.conv_1(input_data)
        out_2 = self.conv_2(out_1)
        out_3 = self.conv_3(out_2)
        output_data = self.conv_4(out_3)
        output_data0 = self.max(out_3)
        out1 = self.layer_last(output_data.view(output_data.size(0), -1))
        out2 = self.layer_second(output_data0.view(output_data0.size(0), -1))#view→reshape
        out = torch.cat((out1, out2), dim=1)  # (batch_size, 256)
        return out

# ConvNet encoder
class EncoderNet(nn.Module):
    def __init__(self, args):
        #
        super(EncoderNet, self).__init__()
        self.args = args

        self.compressNet = CompressNet()
        # todo 随机排列运行分支
        # self.rand_order = SEED_pre.random_1D_seed(self.args.rand_ali_num, self.args.config["eeg_node_num"])
        # print(self.rand_order)

        self.relationAwareness = RelationAwareness()

        # self.convNet = ConvNet(512,self.args)

        self.resNet18_pretrain = Res18()

        self.fcNet = FcNet(int(self.args["num_class"]))

    def forward(self, x):
        out1 = self.compressNet(x)

        # todo 随机排列运行分支

        out2 = self.relationAwareness(out1)

        # out3 = self.convNet(out2)
        out3 = self.resNet18_pretrain(out2)  # pretrained ResNet

        out4 = self.fcNet(out3)

        return out4