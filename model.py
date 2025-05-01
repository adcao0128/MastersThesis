import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool, SAGEConv, GCNConv
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn import global_mean_pool



class GIN(torch.nn.Module):
    def __init__(self, gcn_length, dropout=0.5, hidden=512, train_eps=True, class_num=1):
        super(GIN, self).__init__()
        self.dropout = dropout
        self.gcn_length = gcn_length
        self.train_eps = train_eps
        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(gcn_length, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                # nn.Linear(hidden, hidden),
                # nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                # nn.Linear(hidden, hidden),
                # nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv3 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )

        self.lin1 = nn.Linear(hidden, hidden)
        self.fc1 = nn.Linear(1 * hidden, 1) #clasifier for concat
        self.fc2 = nn.Linear(hidden, 1)   #classifier for inner product



    def reset_parameters(self):

        self.fc1.reset_parameters()

        self.gin_conv1.reset_parameters()
        self.gin_conv2.reset_parameters()
        # self.gin_conv3.reset_parameters()
        self.lin1.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x, edge_index, edge_ids=None, node_pairs=None):
        # GIN encoding
        x = self.gin_conv1(x, edge_index)
        x = self.gin_conv2(x, edge_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Determine which pairs of nodes to score
        if node_pairs is not None:
            node_id = node_pairs.t()  # assume shape [num_pairs, 2]
        elif edge_ids is not None:
            node_id = edge_index[:, edge_ids]
        else:
            raise ValueError("Must provide either edge_ids or node_pairs.")

        x1 = x[node_id[0]]
        x2 = x[node_id[1]]
        x_out = torch.mul(x1, x2)
        x_out = self.fc2(x_out)
        return x_out
    # def forward(self, x, edge_index, train_edge_id):

    #     x = self.gin_conv1(x, edge_index)
    #     x = self.gin_conv2(x, edge_index)
    #     # x = self.gin_conv3(x, edge_index)
    #     x = F.relu(self.lin1(x))
    #     x = F.dropout(x, p=self.dropout, training=self.training)
    #     node_id = edge_index[:, train_edge_id]
    #     x1 = x[node_id[0]]
    #     x2 = x[node_id[1]]
    #     # x = torch.cat([x1, x2], dim=1)
    #     # x = self.fc1(x)
    #     x = torch.mul(x1, x2)
    #     x = self.fc2(x)
        

    #     return x



class GCN(nn.Module):
    def __init__(self, dropout=0.5, hidden=128):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(1280, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.conv4 = GCNConv(hidden, hidden)
  
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.bn4 = nn.BatchNorm1d(hidden)

        self.sag1 = SAGPooling(hidden,0.5)
        self.sag2 = SAGPooling(hidden,0.5)
        self.sag3 = SAGPooling(hidden,0.5)
        self.sag4 = SAGPooling(hidden,0.5)

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)

        self.dropout = nn.Dropout(dropout)
        for param in self.parameters():
            print(type(param), param.size())


    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.fc1(x)
        x = F.relu(x) 
        x = self.bn1(x)
        y = self.sag1(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1] 

        x = self.conv2(x, edge_index)
        x = self.fc2(x)
        x = F.relu(x) 
        x = self.bn2(x)
        y = self.sag2(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]  
        
        x = self.conv3(x, edge_index)
        x = self.fc3(x)
        x = F.relu(x) 
        x = self.bn3(x)
        y = self.sag3(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        x = self.conv4(x, edge_index)
        x = self.fc4(x)
        x = F.relu(x) 
        x = self.bn4(x)
        y = self.sag4(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        # y = self.sag4(x, edge_index, batch = batch)

        return global_mean_pool(y[0], y[3])
        # return y[0]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
class ppi_model(nn.Module):
    def __init__(self, dropout_gcn=0.5, dropout_gin=0.5, hidden_dim_gcn=128, hidden_dim_gin=512):
        super(ppi_model,self).__init__()
        self.BGNN = GCN(dropout_gcn, hidden_dim_gcn)
        self.TGNN = GIN(hidden_dim_gcn, dropout_gin, hidden=hidden_dim_gin)

    def forward(self, batch, p_x_all, p_edge_all, edge_index, train_edge_id, p=0.5):
        edge_index = edge_index.to(device)
        batch = batch.to(torch.int64).to(device)
        x = p_x_all.to(torch.float32).to(device)
        edge = torch.LongTensor(p_edge_all.to(torch.int64)).to(device)
        embs = self.BGNN(x, edge, batch-1)
        final = self.TGNN(embs, edge_index, train_edge_id, p=0.5)
        return final