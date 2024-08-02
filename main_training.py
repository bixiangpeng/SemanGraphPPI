# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2024年05月13日
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

def main(args):
    with open('./data/Annotation_KG/MF_subgraph.pkl', 'rb') as file:
        MF_subgraph = pkl.load(file)
    with open('./data/Annotation_KG/BP_subgraph.pkl', 'rb') as file:
        BP_subgraph = pkl.load(file)
    with open('./data/Annotation_KG/CC_subgraph.pkl', 'rb') as file:
        CC_subgraph = pkl.load(file)
    with open(f'./data/{args.datasetname}/Interaction_KG/IKG_edge.pkl', 'rb') as file:
        IKG_edge = pkl.load(file)
    with open(f'./data/{args.datasetname}/Interaction_KG/index_map_dict.pkl', 'rb') as file:
        index_map = pkl.load(file)
    with open(f'./data/{args.datasetname}/Interaction_KG/edge_index_map_dict.pkl', 'rb') as file:
        edge_index_map = pkl.load(file)
    with open(f'./data/{args.datasetname}/Interaction_KG/annotation_index_map.pkl', 'rb') as file:
        annotation_index_map = pkl.load(file)
    with open(f'./data/{args.datasetname}/Interaction_KG/annotation_batch.pkl', 'rb') as file:
        annotation_batch = pkl.load(file)

    go_graph = (MF_subgraph, BP_subgraph, CC_subgraph)
    device = torch.device('cuda:' + str(args.device_id) if torch.cuda.is_available() else "cpu")
    IKG_edge = IKG_edge.to(device)

    train_pns = []
    with open(f'./data/{args.datasetname}/train.tsv', 'r') as fh:
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')
            words = re.split('  |\t', line)
            train_pns.append((words[0], words[1], int(words[2])))

    test_pns = []
    with open(f'./data/{args.datasetname}/test.tsv', 'r') as fh:
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')
            words = re.split('  |\t', line)
            test_pns.append((words[0], words[1], int(words[2])))

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
            torch.save(model.state_dict(), f'./model_pkl/{args.datasetname}/' + f'epoch' + f'{epoch}.pkl')
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

        with open(f'./model_pkl/{args.datasetname}/results.txt', 'a+') as fp:
            fp.write('epoch:' + str(epoch + 1) + '\ttrainacc=' + str(acc) + '\ttrainloss=' + str(avg_loss.item()) + '\tacc=' + str(test_acc) + '\tprec=' + str(test_prec) + '\trecall=' + str(test_recall) + '\tf1=' + str(test_f1) + '\tauc=' + str(test_auc) + '\tspec=' + str(test_spec) + '\tmcc=' + str(test_mcc) + '\tlr=' + str(optimizer.param_groups[0]['lr'])+'\n')


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--datasetname', type = str , default = 'DIP_S.cerevisiae',choices=['DIP_S.cerevisiae', 'STRING_H.sapiens', 'STRING_S.cerevisiae'])
    parse.add_argument('--device_id', type = int, default = 0)
    parse.add_argument('--batch_size', type=int, default = 32)
    parse.add_argument('--epochs', type=int, default = 50)
    parse.add_argument('--lr', type=float, default=0.001)
    parse.add_argument('--num_workers', type=int, default= 8)
    parse.add_argument('--do_Save', type=bool, default=True)
    parse.add_argument('--super_ratio', type=float, default = 0.2)
    parse.add_argument('--layers', type=int, default= 8)
    parse.add_argument('--hidden_dim', type=int, default= 64)
    args = parse.parse_args()

    main(args)