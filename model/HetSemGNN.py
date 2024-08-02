import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv
from .utils import masked_edge_index, SemanticsAttention

class HetSemGCN(nn.Module):
    def __init__(self, input_dim = 64, hidden_dim = 64, num_relations = 5):
        super(HetSemGCN, self).__init__()
        self.num_realtions = num_relations
        self.GCN_list = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_relations)])
        self.semantic_attn = SemanticsAttention(hidden_dim, 0)

    def forward(self, x, edge_index, edge_type):
        out = []
        for i in range(self.num_realtions):
            new_edge_index = masked_edge_index(edge_index, edge_type == i)
            temp_x = self.GCN_list[i](x, new_edge_index)
            out.append(temp_x)
        x = self.semantic_attn(out)
        return x


