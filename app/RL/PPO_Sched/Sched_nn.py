import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical

from torch_geometric.nn import GCNConv, SAGEConv, TopKPooling, global_mean_pool as gap, global_max_pool as gmp, MLP
from torch_geometric.nn.models.basic_gnn import GCN
from torch_geometric.data import Batch
from torch_geometric.utils import softmax

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.num_layers = 3
        self.out_channels = args.hidden_width
        self.GCN = GCN(in_channels=args.state_dim, 
                       hidden_channels=args.hidden_width,
                       num_layers=self.num_layers,
                       out_channels=self.out_channels,
                       dropout=0.5,
                       act = "relu", # tanh
                       act_first = False,
                       norm = "LayerNorm", # LayerNorm
                       jk = "cat")
        self.fc1 = nn.Linear(args.hidden_width, args.hidden_width // 2)
        self.fc2 = nn.Linear(args.hidden_width // 2, 1)

        self.GCN.reset_parameters()
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)

    def forward(self, s):
        x, edge = s.x, s.edge_index
        batch, mask = s.batch, s.mask
        if batch is None:
            x = self.GCN(x, edge)
            x = F.tanh(self.fc1(x))
            x = self.fc2(x)
            x = x + mask.float() * -1e9
            prob = F.softmax(x, dim=0).squeeze(1)
            return prob
        else:
            batch_size = s.batch_size
            x = self.GCN(x, edge, batch = batch)
            x = F.tanh(self.fc1(x))
            x = self.fc2(x)
            x = x + mask.float() * -1e9
            prob = softmax(x, batch, dim=0).view(batch_size, -1)
            return prob
    
    @torch.no_grad()
    def inference(self, s):
        return self.GCN.inference(s)
    
# class Critic(nn.Module):
#     def __init__(self, args):
#         super(Critic, self).__init__()        
#         self.conv1 = GCNConv(args.state_dim, args.hidden_width)
#         self.conv2 = GCNConv(args.hidden_width, args.hidden_width)
#         self.conv3 = GCNConv(args.hidden_width, args.hidden_width)
#         self.fc1 = torch.nn.Linear(args.hidden_width * 2, args.hidden_width)
#         self.fc2 = torch.nn.Linear(args.hidden_width, 1)    
#         self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh    

#         if args.use_orthogonal_init:
#             print("------use_orthogonal_init------")
#             orthogonal_init(self.fc1)
#             orthogonal_init(self.fc2)
#         self.conv1.reset_parameters()
#         self.conv2.reset_parameters()
#         self.conv3.reset_parameters()

#     def forward(self, s):
#         x, edge_index, batch, batch_size = s.x, s.edge_index, s.batch, s.batch_size

#         x = self.activate_func(self.conv1(x, edge_index))
#         # x = F.dropout(x, p=0.5, training=self.training)
#         x1 = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)

#         x = self.activate_func(self.conv2(x, edge_index))
#         # x = F.dropout(x, p=0.5, training=self.training)
#         x2 = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)

#         x = self.activate_func(self.conv3(x, edge_index))
#         # x = F.dropout(x, p=0.5, training=self.training)
#         x3 = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)

#         x = x1 + x2 + x3
#         x = self.activate_func(self.fc1(x))
#         return self.fc2(x)
    
class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.num_layers = 3
        self.out_channels = args.hidden_width
        self.GCN = GCN(in_channels=args.state_dim, 
                       hidden_channels=args.hidden_width,
                       num_layers=self.num_layers,
                       out_channels=self.out_channels,
                       dropout=0.5,
                       act = "tanh", # tanh
                       act_first = False,
                       norm = "LayerNorm", # LayerNorm
                       jk = "cat")
        self.fc1 = nn.Linear(args.hidden_width, args.hidden_width // 2)
        self.fc2 = nn.Linear(args.hidden_width // 2, 1)

        self.GCN.reset_parameters()
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)

    def forward(self, s):
        x, edge = s.x, s.edge_index
        batch, mask = s.batch, s.mask
        if batch is None:
            x = self.GCN(x, edge)
            x = F.tanh(self.fc1(x))
            x = self.fc2(x)
            return x[-1]
        else:
            batch_size = s.batch_size
            x = self.GCN(x, edge, batch = batch)
            x = F.tanh(self.fc1(x))
            x = self.fc2(x)
            _, last_indices = torch.unique_consecutive(batch, return_inverse=False, return_counts=True)
            last_indices = last_indices.cumsum(0) - 1 
            return x[last_indices]