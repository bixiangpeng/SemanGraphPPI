# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2024年07月20日
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
    with open(f'./data/{args.datasetname}/Interaction_KG/annotation_index_map.pkl', 'rb') as file:
        annotation_index_map = pkl.load(file)
    with open(f'./data/{args.datasetname}/Interaction_KG/annotation_batch.pkl', 'rb') as file:
        annotation_batch = pkl.load(file)

    go_graph = (MF_subgraph, BP_subgraph, CC_subgraph)
    device = torch.device('cuda:' + str(args.device_id) if torch.cuda.is_available() else "cpu")
    IKG_edge = IKG_edge.to(device)

    test_pns = []
    with open(f'./data/{args.datasetname}/test.tsv', 'r') as fh:
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')
            words = re.split('  |\t', line)
            test_pns.append((words[0], words[1], int(words[2])))
    test_dataset = Dataset_test(index_map_dict = index_map,  pns=test_pns)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle= False, collate_fn = collate_test, num_workers = args.num_workers )

    model = SemanGraphPPI(num_clusters=int(2497 * args.super_ratio), num_layers=args.layers, hidden_dim=args.hidden_dim)
    model = model.to(device)


    path = f'./model_pkl/{args.datasetname}/pretrained_model.pkl'
    model.load_state_dict(torch.load(path),strict=False)
    model.eval()

    total_pred_values = torch.Tensor()
    total_pred_labels = torch.Tensor()
    total_true_labels = torch.Tensor()
    go_graph_list = []
    for index, graph in enumerate(go_graph):
        go_graph_list.append((graph[0], graph[1].to(device), graph[2].to(device), graph[3].to(device)))
    print('Make prediction for {} samples...'.format(len(test_loader.dataset)))
    with torch.no_grad():
        for batch_idx, (pid1, pid2, y) in enumerate(test_loader):
            pid1, pid2 = pid1.to(device), pid2.to(device)
            annotation_index_map = annotation_index_map.to(device)
            annotation_batch = annotation_batch.to(device)
            output = model(IKG_edge, go_graph_list, annotation_index_map, annotation_batch, pid1, pid2, None)
            predicted_values = torch.sigmoid(output)
            predicted_labels = torch.round(predicted_values)
            total_pred_values = torch.cat((total_pred_values, predicted_values.cpu()), 0)  # predicted values
            total_pred_labels = torch.cat((total_pred_labels, predicted_labels.cpu()), 0)  # predicted labels
            total_true_labels = torch.cat((total_true_labels, y.view(-1, 1).cpu()), 0)  # ground truth

        G, P_value, P_label = total_true_labels.numpy().flatten(), total_pred_values.numpy(), total_pred_labels.numpy().flatten()
        test_acc = accuracy_score(G,P_label)
        test_prec = precision_score(G, P_label)
        test_recall = recall_score(G, P_label)
        test_f1 = f1_score(G, P_label)
        test_auc = roc_auc_score(G, P_value)
        con_matrix = confusion_matrix(G, P_label)
        test_spec = con_matrix[0][0] / ( con_matrix[0][0] + con_matrix[0][1] )
        test_mcc = ( con_matrix[0][0] * con_matrix[1][1] - con_matrix[0][1] * con_matrix[1][0] ) / (((con_matrix[1][1] +con_matrix[0][1]) * (con_matrix[1][1] +con_matrix[1][0]) * (con_matrix[0][0] +con_matrix[0][1]) * (con_matrix[0][0] +con_matrix[1][0])) ** 0.5)

        print("acc: ", test_acc, " ; prec: ", test_prec, " ; recall: ", test_recall, " ; f1: ", test_f1, " ; auc: ", test_auc, " ; spec:", test_spec, " ; mcc: ", test_mcc)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--datasetname', type = str , default = 'DIP_S.cerevisiae',choices=['DIP_S.cerevisiae', 'STRING_H.sapiens', 'STRING_S.cerevisiae'])
    parse.add_argument('--device_id', type = int, default = 0)
    parse.add_argument('--batch_size', type=int, default = 512)
    parse.add_argument('--lr', type=float, default=0.001)
    parse.add_argument('--num_workers', type=int, default= 8)
    parse.add_argument('--super_ratio', type=float, default = 0.2)
    parse.add_argument('--layers', type=int, default= 8)
    parse.add_argument('--hidden_dim', type=int, default= 64)
    args = parse.parse_args()

    main(args)