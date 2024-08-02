# -*- coding:utf-8 -*-
"""
作者：毕宸浩
日期：2023年07月25日
"""
import torch
from tqdm import tqdm

def test(model, device, loader, IKG_edge, annotation_graph, annotation_index_map, annotation_batch, args):
    model.eval()
    total_pred_values = torch.Tensor()
    total_pred_labels = torch.Tensor()
    total_true_labels = torch.Tensor()
    go_graph_list = []
    for index, graph in enumerate(annotation_graph):
        go_graph_list.append((graph[0], graph[1].to(device), graph[2].to(device), graph[3].to(device)))
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, (pid1, pid2, y) in tqdm(enumerate(loader), total=len(loader)):
            pid1, pid2 = pid1.to(device), pid2.to(device)
            annotation_index_map = annotation_index_map.to(device)
            annotation_batch = annotation_batch.to(device)
            output = model(IKG_edge, go_graph_list, annotation_index_map, annotation_batch, pid1, pid2, None)
            predicted_values = torch.sigmoid(output)
            predicted_labels = torch.round(predicted_values)
            total_pred_values = torch.cat((total_pred_values, predicted_values.cpu()), 0)  # predicted values
            total_pred_labels = torch.cat((total_pred_labels, predicted_labels.cpu()), 0)  # predicted labels
            total_true_labels = torch.cat((total_true_labels, y.view(-1, 1).cpu()), 0)  # ground truth
    return total_true_labels.numpy().flatten(), total_pred_values.numpy(), total_pred_labels.numpy().flatten()

def train(model, train_loader, IKG_edge, annotation_graph, annotation_index_map, annotation_batch, device, optimizer, criterion,args):
    model.train()
    total_loss = 0
    n_batches = 0
    correct = 0
    go_graph_list = []
    for index, graph in enumerate(annotation_graph):
        go_graph_list.append((graph[0], graph[1].to(device), graph[2].to(device), graph[3].to(device)))
    for batch_idx,(pid1, pid2,  y, edge_index_map)in tqdm(enumerate(train_loader),total = len(train_loader)):
        pid1, pid2 = pid1.to(device), pid2.to(device)
        edge_index_map = edge_index_map.to(device)
        annotation_index_map = annotation_index_map.to(device)
        annotation_batch = annotation_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(IKG_edge, go_graph_list, annotation_index_map, annotation_batch, pid1, pid2, edge_index_map)
        loss = criterion(y_pred, y.view(-1, 1).float().to(device), )
        y_pred = torch.round(torch.sigmoid(y_pred).squeeze(1))
        correct += torch.eq(y_pred.cpu(),y).data.sum()
        total_loss += loss.data
        loss.backward()
        optimizer.step()
        n_batches += 1
    avg_loss = total_loss / n_batches
    acc = correct.cpu().numpy() / (len(train_loader.dataset))
    print(f"train avg_loss is {avg_loss}")
    print("train ACC = ", acc)
    return avg_loss, acc