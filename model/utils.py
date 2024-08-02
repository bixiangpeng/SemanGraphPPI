# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2024年07月07日
"""
import torch.nn as nn
import torch
from torch_sparse import  masked_select_nnz
from torch import Tensor
import torch.nn.functional as F
from torch_scatter import scatter_softmax
from torch_geometric.nn import global_add_pool as gap

def masked_edge_index(edge_index, edge_mask: Tensor):
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    return masked_select_nnz(edge_index, edge_mask, layout='coo')

class SemanticsAttention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(SemanticsAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax(dim = 0)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i] * beta[i]
        return z_mp


class Attention_pooling(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention_pooling, self).__init__()
        self.negative_slope = 0.2
        self.a = nn.Parameter(torch.empty(size=(hidden_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # Xavier均匀分布初始化

    def forward(self, feature, batch):
        att_logits = F.leaky_relu(torch.matmul(feature, self.a), self.negative_slope)
        att_score = scatter_softmax(att_logits, batch,dim=0).expand(-1, feature.shape[1])
        output = att_score * feature
        output = gap(output, batch)
        return output
