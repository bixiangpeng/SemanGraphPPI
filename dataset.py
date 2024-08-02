# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年07月24日
"""
from torch.utils.data import Dataset
import torch


def collate_train(data):
    pid1_list = [ item[0] for item in data]
    pid2_list = [item[1] for item in data]
    label_list = [item[2] for item in data]
    edge_index_list = [item[3] for item in data if item[3] != -1] + [item[4] for item in data if item[4] != -1]
    return torch.LongTensor(pid1_list), torch.LongTensor(pid2_list), torch.tensor(label_list), torch.LongTensor(edge_index_list)
def collate_test(data):
    pid1_list = [ item[0] for item in data]
    pid2_list = [item[1] for item in data]
    label_list = [item[2] for item in data]
    return torch.LongTensor(pid1_list), torch.LongTensor(pid2_list), torch.tensor(label_list)


class Dataset_train(Dataset):
    def __init__(self,  index_map_dict = None,edge_index_map_dict = None, pns= None):
        super(Dataset_train,self).__init__()
        self.pns = pns
        self.index_map_dict = index_map_dict
        self.edge_index_map_dict = edge_index_map_dict

    def __len__(self):
        return len(self.pns)

    def __getitem__(self, index):
        protein1, protein2, label = self.pns[index]
        pid1, pid2 = self.index_map_dict[protein1], self.index_map_dict[protein2]
        if label == 1:
            edge_index_map1 = self.edge_index_map_dict[f'{pid1}-{pid2}']
            edge_index_map2 = self.edge_index_map_dict[f'{pid2}-{pid1}']
        else:
            edge_index_map1 = -1
            edge_index_map2 = -1
        return pid1, pid2, label, edge_index_map1, edge_index_map2

class Dataset_test(Dataset):
    def __init__(self,  index_map_dict = None, pns= None):
        super(Dataset_test,self).__init__()
        self.pns = pns
        self.index_map_dict = index_map_dict

    def __len__(self):
        return len(self.pns)

    def __getitem__(self, index):

        protein1, protein2, label = self.pns[index]
        pid1, pid2 = self.index_map_dict[protein1], self.index_map_dict[protein2]
        return pid1, pid2, label