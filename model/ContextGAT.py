# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2024年07月06日
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class ContextGAT(MessagePassing):
    def __init__(self, input_dim, output_dim):
        super(ContextGAT, self).__init__(aggr='add')
        self.dropout = 0
        self.output_dim = output_dim
        self.a = nn.Parameter(torch.zeros(size=(3 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, edge_index, context):
        x_ = self.linear(x)
        row, col = edge_index
        a_input = torch.cat([x_[row], x_[col]], dim=-1)
        a_input = torch.cat([a_input, context - x_[row] - x_[col] ], dim=-1)
        temp = torch.matmul(a_input, self.a).squeeze(-1)
        alpha = F.leaky_relu(temp, 0.2)
        alpha = softmax(alpha, col, dim=-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = self.propagate(edge_index, x = x_, alpha = alpha)
        return out

    def message(self, x_j, alpha):
        return alpha.view(-1, 1) * x_j

