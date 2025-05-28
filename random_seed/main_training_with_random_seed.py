# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2025年05月20日
"""

import pickle as pkl
import argparse
import re
from dataset import Dataset_train, Dataset_test, collate_train, collate_test
from torch.utils.data import DataLoader
from model.SemanGraphPPI import SemanGraphPPI
import torch
from train_and_test import train, test
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.metrics import confusion_matrix

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def load_data(train_pns, args):
    with open(f'./data/{args.datasetname}/seed/index_map_dict.pkl', 'rb') as f:
        index_map_dict = pkl.load(f)

    ppi_adj = np.zeros((len(index_map_dict), len(index_map_dict)), dtype=int)
    for temp_pair in train_pns:
        pid1, pid2, label = temp_pair
        if label == 1:
            ppi_adj[index_map_dict[pid1]][index_map_dict[pid2]] = 1
            ppi_adj[index_map_dict[pid2]][index_map_dict[pid1]] = 1

    np.fill_diagonal(ppi_adj, 0)
    ppi_adj = (ppi_adj + np.eye(len(index_map_dict))).astype(int)
    ppi_adj = np.argwhere(ppi_adj == 1)
    edge_index_map_dict = {}
    for index, edge in enumerate(ppi_adj):
        edge_index_map_dict[f'{edge[0]}-{edge[1]}'] = index
    IKG_edge = torch.LongTensor(ppi_adj.transpose(1, 0))

    with open(f'./data/{args.datasetname}/seed/go_id_dict.pkl', 'rb') as file:
        go_to_index_dict = pkl.load(file)
    with open(f'./data/{args.datasetname}/seed/string_to_go.pkl','rb') as file:
        pro_to_GoID = pkl.load(file)

    go_index_map = []
    go_batch = []
    no_hit_total = []
    for index, pro in enumerate(list(index_map_dict.keys())):
        Go_list = pro_to_GoID[pro]
        if len(Go_list)== 0:
            go_index_map.append(42255)
            go_batch.append(index)
        else:
            no_hit = []
            for go_term_id in Go_list:
                if go_term_id in go_to_index_dict.keys():
                    go_index_map.append(go_to_index_dict[go_term_id])
                    go_batch.append(index)
                else:
                    no_hit.append(go_term_id)
                    no_hit_total.append(go_term_id)
                    continue
            if len(no_hit) == len(Go_list):
                print('prot have no good goterm!')
                go_index_map.append(42255)
                go_batch.append(index)
    no_hit_total = list(set(no_hit_total))
    print(f'{len(no_hit_total)} goterms have not been mapped!')
    annotation_index_map = torch.tensor(go_index_map, dtype=torch.long)
    annotation_batch = torch.tensor(go_batch, dtype=torch.long)

    return IKG_edge, index_map_dict, edge_index_map_dict, annotation_index_map, annotation_batch


def main(args):
    with open('./data/Annotation_KG/MF_subgraph.pkl', 'rb') as file:
        MF_subgraph = pkl.load(file)
    with open('./data/Annotation_KG/BP_subgraph.pkl', 'rb') as file:
        BP_subgraph = pkl.load(file)
    with open('./data/Annotation_KG/CC_subgraph.pkl', 'rb') as file:
        CC_subgraph = pkl.load(file)
    go_graph = (MF_subgraph, BP_subgraph, CC_subgraph)
    device = torch.device('cuda:' + str(args.device_id) if torch.cuda.is_available() else "cpu")

    #### Replace the dataset partitioning with a random seed-based splitting method
    df_total = pd.read_csv(f'./data/{args.datasetname}/total_any.tsv', header=None, sep='\t')
    new_df_train, new_df_test = train_test_split(df_total, test_size=0.2, random_state=42)

    train_pns = []
    for index, row in new_df_train.iterrows():
        train_pns.append((row[0], row[1], int(row[2])))

    test_pns = []
    for index, row in new_df_test.iterrows():
        test_pns.append((row[0], row[1], int(row[2])))

    IKG_edge, index_map, edge_index_map, annotation_index_map, annotation_batch = load_data(train_pns, args)
    IKG_edge = IKG_edge.to(device)

    train_dataset = Dataset_train(index_map_dict = index_map, edge_index_map_dict = edge_index_map, pns = train_pns)
    train_loader = DataLoader(dataset=train_dataset, batch_size = args.batch_size , shuffle = True, collate_fn = collate_train, num_workers = args.num_workers)
    test_dataset = Dataset_test(index_map_dict = index_map, pns=test_pns)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle= False, collate_fn = collate_test, num_workers = args.num_workers )

    model = SemanGraphPPI(num_clusters=int(2497 * args.super_ratio), num_layers=args.layers, hidden_dim=args.hidden_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        print("Running EPOCH", epoch + 1)
        avg_loss, acc = train(model, train_loader, IKG_edge, go_graph, annotation_index_map, annotation_batch, device, optimizer, criterion, args)
        if (args.do_Save):
            torch.save(model.state_dict(), f'./model_pkl/{args.datasetname}/seed/' + f'epoch' + f'{epoch}.pkl')
        G, P_value, P_label = test(model, device, test_loader, IKG_edge, go_graph, annotation_index_map, annotation_batch, args)

        test_acc = accuracy_score(G,P_label)
        test_prec = precision_score(G, P_label)
        test_recall = recall_score(G, P_label)
        test_f1 = f1_score(G, P_label)
        test_auc = roc_auc_score(G, P_value)
        con_matrix = confusion_matrix(G, P_label)
        test_spec = con_matrix[0][0] / ( con_matrix[0][0] + con_matrix[0][1] )
        test_mcc = ( con_matrix[0][0] * con_matrix[1][1] - con_matrix[0][1] * con_matrix[1][0] ) / (((con_matrix[1][1] +con_matrix[0][1]) * (con_matrix[1][1] +con_matrix[1][0]) * (con_matrix[0][0] +con_matrix[0][1]) * (con_matrix[0][0] +con_matrix[1][0])) ** 0.5)

        print("acc: ", test_acc, " ; prec: ", test_prec, " ; recall: ", test_recall, " ; f1: ", test_f1, " ; auc: ",
              test_auc, " ; spec:", test_spec, " ; mcc: ", test_mcc,'lr:',optimizer.param_groups[0]['lr'])

        with open(f'./model_pkl/{args.datasetname}/seed/results.txt', 'a+') as fp:
            fp.write('epoch:' + str(epoch + 1) + '\ttrainacc=' + str(acc) + '\ttrainloss=' + str(avg_loss.item()) + '\tacc=' + str(test_acc) + '\tprec=' + str(test_prec) + '\trecall=' + str(test_recall) + '\tf1=' + str(test_f1) + '\tauc=' + str(test_auc) + '\tspec=' + str(test_spec) + '\tmcc=' + str(test_mcc) + '\tlr=' + str(optimizer.param_groups[0]['lr'])+'\n')


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--datasetname', type = str , default = 'DIP_S.cerevisiae')
    parse.add_argument('--mode', type = str , default = 'any')
    parse.add_argument('--device_id', type = int, default = 0)
    parse.add_argument('--batch_size', type=int, default = 32)
    parse.add_argument('--epochs', type=int, default = 50)
    parse.add_argument('--lr', type=float, default=0.001)
    parse.add_argument('--num_workers', type=int, default= 2)
    parse.add_argument('--do_Save', type=bool, default=True)
    parse.add_argument('--super_ratio', type=float, default = 0.2)
    parse.add_argument('--layers', type=int, default= 8)
    parse.add_argument('--hidden_dim', type=int, default= 64)
    args = parse.parse_args()

    main(args)