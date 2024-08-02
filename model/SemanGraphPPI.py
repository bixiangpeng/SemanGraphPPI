import torch.nn as nn
from torch_geometric.nn.conv import GCNConv
from .HetSemGNN import HetSemGCN
from .ContextGAT import ContextGAT
import torch
from torch_geometric.utils import index_to_mask
from .utils import Attention_pooling

class IKG_Enhancer(torch.nn.Module):
    def __init__(self, hidden_dim = 64, num_clusters = 10):
        super(IKG_Enhancer, self).__init__()
        self.relu = nn.ReLU()
        self.GCN_K = GCNConv(hidden_dim, hidden_dim)
        self.GCN_V = GCNConv(hidden_dim, hidden_dim)
        self.Super_Q = torch.nn.Parameter(torch.Tensor(num_clusters, hidden_dim))
        nn.init.xavier_normal_(self.Super_Q.data, gain=1.414)
        self.MHA = torch.nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, dropout=0.2)
        self.contextGAT = ContextGAT(input_dim = hidden_dim, output_dim = hidden_dim)
        self.transform1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.transform2 = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x_K = self.GCN_K(x, edge_index)
        x_V = self.GCN_V(x, edge_index)
        Cluster_X = self.MHA(self.Super_Q, x_K, x_V, attn_mask=None, key_padding_mask=None, need_weights=None)[0]
        context = torch.mean(Cluster_X, dim=0)
        x = self.contextGAT(x_V, edge_index, context)
        x = self.relu(self.transform1(x))
        x = self.relu(self.transform2(x))
        return x

class Semantic_Encoder(nn.Module):
    def __init__(self, hidden_dim = 64, num_relations = 5, num_layers = 10):
        super(Semantic_Encoder, self).__init__()
        self.relu = nn.ReLU()
        self.num_layers = num_layers
        self.HSGNN_list = nn.ModuleList([HetSemGCN(hidden_dim, hidden_dim, num_relations=num_relations) for _ in range(self.num_layers)])
        self.LN_list = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.num_layers)])

    def forward(self, annotation_graph):
        _, annotation_feature, annotation_edges, annotation_edges_type = annotation_graph
        for i in range(self.num_layers):
            annotation_feature_ = self.relu(self.HSGNN_list[i](annotation_feature, annotation_edges, annotation_edges_type))
            annotation_feature = self.LN_list[i](annotation_feature_ + annotation_feature)
        return annotation_feature



class SemanGraphPPI(nn.Module):
    def __init__(self,input_dim = 64, hidden_dim = 64, output_dim = 1, num_clusters = 10, num_layers = 10):
        super(SemanGraphPPI, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.MF_encoder = Semantic_Encoder(hidden_dim = hidden_dim, num_relations = 5, num_layers = num_layers)
        self.BP_encoder = Semantic_Encoder(hidden_dim = hidden_dim, num_relations = 5, num_layers = num_layers)
        self.CC_encoder = Semantic_Encoder(hidden_dim = hidden_dim, num_relations = 5, num_layers = num_layers)
        self.attention_pooling = Attention_pooling(hidden_dim)
        self.IKG_enhancer = IKG_Enhancer(hidden_dim = hidden_dim, num_clusters = num_clusters)

        self.fc1 = nn.Linear(hidden_dim * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, output_dim)

    def mask_edge(self, ppi_edge, edge_index_mask):
        new_ppi_edge = ppi_edge.cpu().to(edge_index_mask.device)
        row, col = new_ppi_edge[0], ppi_edge[1]
        new_row, new_col = row[edge_index_mask], col[edge_index_mask]
        return torch.stack([new_row, new_col], dim=0).to(edge_index_mask.device)

    def forward(self, IKG_edge, annotation_KG, annotation_index_map, annotation_batch, pid1, pid2, edge_index_map):
        MF_subgraph, BP_subgraph, CC_subgraph = annotation_KG[0], annotation_KG[1], annotation_KG[2]
        MF_feature = self.MF_encoder(MF_subgraph)
        BP_feature = self.BP_encoder(BP_subgraph)
        CC_feature = self.CC_encoder(CC_subgraph)
        embedding_voca = torch.cat([MF_feature, BP_feature, CC_feature], dim=0)
        temp_features = embedding_voca[annotation_index_map]
        protein_feature = self.attention_pooling(temp_features, annotation_batch)

        if self.training:
            edge_index_mask = ~index_to_mask(edge_index_map, size=IKG_edge.shape[1])
            edge_index = self.mask_edge(IKG_edge, edge_index_mask)
        else:
            edge_index = IKG_edge

        H_ppi = self.IKG_enhancer(protein_feature, edge_index)
        h1_feat = H_ppi[pid1]
        h2_feat = H_ppi[pid2]

        hc = torch.cat([h1_feat, h2_feat], dim=1)
        hc = self.fc1(hc)
        hc = self.relu(hc)
        hc = self.dropout(hc)
        hc = self.fc2(hc)
        hc = self.relu(hc)
        hc = self.dropout(hc)
        output = self.out(hc)
        return output