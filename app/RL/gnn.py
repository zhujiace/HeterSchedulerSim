import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, SAGEConv, TopKPooling, global_mean_pool as gap, global_max_pool as gmp, MLP
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot_orthogonal, glorot
import torch.nn as nn
import torch.nn.functional as F
from fused_graph import create_fused_graph, choose_node

class ModelParams:
    def __init__(self, input_dim, embed_dim, hidden_dim, is_sampling, temperature=1.0, decay_rate=0.99):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.is_sampling = is_sampling
        self.temperature = temperature
        self.decay_rate = decay_rate

# class SchedulerGNN(torch.nn.Module):
#     def __init__(self, model_params:ModelParams):
#         super(SchedulerGNN, self).__init__()
#         self.input_dim = model_params.input_dim
#         self.hidden_dim = model_params.hidden_dim

#         self.conv1 = GCNConv(self.input_dim, self.hidden_dim)
#         self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)

#         # Multi-layer perceptron (MLP) for better feature extraction
#         self.mlp = torch.nn.Sequential(
#             torch.nn.Linear(self.hidden_dim * 3, self.hidden_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(self.hidden_dim, self.hidden_dim),
#             torch.nn.ReLU(),
#             torch.nn.Linear(self.hidden_dim, 1)
#         )

#         self.is_sampling = model_params.is_sampling
#         self.temperature = model_params.temperature
#         self.decay_rate = model_params.decay_rate
#         self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#     def forward(self, data):
#         # Get the input data
#         x, edge_index = data.x, data.edge_index
#         dag_summary_index = data.dag_summary_index
#         request = data.request

#         affinity_flags = data.x[:, 0]
#         current_proc_flags = data.x[:, 1]
#         schedulable_flags = data.x[:, 2]
#         affinity_request = request[0]
#         dag_idx = 0
#         node_index_offset = [0]
#         if isinstance(dag_summary_index[0], list):
#             dag_summary_index = dag_summary_index[0]
#         for index in dag_summary_index:
#             node_index_offset.append(index + 1)
            
#         # Graph Convolutional layers for feature propagation
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         x = F.relu(x)

#         # Generate scheduling scores for all nodes (only for schedulable nodes)
#         node_scores = []
#         node_candidates = []
#         for i in range(x.size(0) - 1):  # Iterate over all nodes
#             if (i not in dag_summary_index) and (affinity_flags[i] == affinity_request) and (schedulable_flags[i] == 1) and (current_proc_flags[i] == -1):
#                 node_embedding = x[i]
#                 while(i > dag_summary_index[dag_idx]):
#                     dag_idx += 1
#                 dag_emb = x[dag_summary_index[dag_idx]]
#                 global_emb = x[-1]
#                 combined_input = torch.cat([node_embedding, dag_emb, global_emb], dim=0)  # Concatenate embeddings

#                 # Pass through MLP to better extract features
#                 score = self.mlp(combined_input)  # Use MLP to generate a better score for this node
#                 node_scores.append(score)
#                 node_candidates.append([dag_idx, i - node_index_offset[dag_idx], score])
#                 # print(f"dag{dag_idx}, node{i-node_index_offset}, score: {score}")

#         if len(node_scores) > 0:
#             # Stack the scores for all nodes
#             node_scores = torch.stack(node_scores, dim=0).squeeze(-1)  # Shape (num_nodes,)

#             # Apply softmax to the scores for all schedulable nodes
#             selection_probs = F.softmax(node_scores/self.temperature, dim=0)
#             # print(node_candidates)
#             if self.is_sampling:
#                 selected_node = torch.multinomial(selection_probs, 1)
#                 return node_candidates[selected_node], selected_node, selection_probs
#             else:
#                 selected_node = torch.argmax(selection_probs)
#                 return node_candidates[selected_node], selected_node, selection_probs
#         else:
#             return -1, -1, -1

#     def update_temperature(self):
#         self.temperature *= self.decay_rate

class _SchedulerGNN(torch.nn.Module):
    def __init__(self, model_params:ModelParams):
        super(_SchedulerGNN, self).__init__()
        
        self.input_dim = model_params.input_dim
        self.embed_dim = model_params.embed_dim
        self.hidden_dim = model_params.hidden_dim

        self.node_embedding = MLP([self.input_dim, self.embed_dim])
        self.conv1 = GCNConv(self.embed_dim, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.conv3 = GCNConv(self.hidden_dim, self.hidden_dim)
        # self.mlp = MLP([self.hidden_dim*3, self.hidden_dim, 1], dropout=[0.5, 0])
        self.mlp = MLP([self.hidden_dim*3, self.hidden_dim, 1])

        self.activate_func = nn.ReLU()
        # self.activate_func = nn.Tanh()

    def forward(self, data):
        x, edge_index, batch, batch_size = data.x, data.edge_index, data.batch, data.batch_size
        request, timestamp = data.request, data.timestamp
        affinity_flags = x[:, 0]
        current_proc_flags = x[:, 1]
        schedulable_flags = x[:, 2]
        affinity_request = request.view(batch_size,-1)[:, 0][batch]
        timestamp = timestamp.view(batch_size,-1)[batch]
        
        mask = data.mask
        
        if mask.float().sum() == mask.size(0):
            return None


        x = torch.cat([x, affinity_request.unsqueeze(1), timestamp], dim=1)
        x = self.node_embedding(x, batch, batch_size)

        x = self.activate_func(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0.5, training=self.training)
        x1 = torch.cat([x, gap(x, batch)[batch], gmp(x, batch)[batch]], dim=1)

        x = self.activate_func(self.conv2(x, edge_index))
        # x = F.dropout(x, p=0.5, training=self.training)
        x2 = torch.cat([x, gap(x, batch)[batch], gmp(x, batch)[batch]], dim=1)

        x = self.activate_func(self.conv3(x, edge_index))
        # x = F.dropout(x, p=0.5, training=self.training)
        x3 = torch.cat([x, gap(x, batch)[batch], gmp(x, batch)[batch]], dim=1)
        
        x = x1 + x2 + x3

        x = self.mlp(x, batch, batch_size)

        # x = x.masked_fill(mask, float('-inf'))
        x = x + mask.float() * -1e9
        x = softmax(x, batch).squeeze(1).view(batch_size, -1)

        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.node_embedding.reset_parameters()
        self.mlp.reset_parameters()

class ValueGNN(torch.nn.Module):
    def __init__(self, model_params:ModelParams):
        super(ValueGNN, self).__init__()
        self.input_dim = model_params.input_dim
        self.embed_dim = model_params.embed_dim
        self.hidden_dim = model_params.hidden_dim
        
        self.node_embedding = MLP([self.input_dim, self.embed_dim])
        self.conv1 = GCNConv(self.embed_dim, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.conv3 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.fc1 = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, 1)

        self.activate_func = nn.ReLU()
        # self.activate_func = nn.Tanh()

    def forward(self, data:Batch):
        x, edge_index, batch, batch_size = data.x, data.edge_index, data.batch, data.batch_size
        request, timestamp = data.request, data.timestamp
        affinity_request = request.view(batch_size,-1)[:, 0][batch]
        timestamp = timestamp.view(batch_size,-1)[batch]
        x = torch.cat([x, affinity_request.unsqueeze(1), timestamp], dim=1)
        x = self.node_embedding(x, batch, batch_size)

        x = self.activate_func(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0.5, training=self.training)
        x1 = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)

        x = self.activate_func(self.conv2(x, edge_index))
        # x = F.dropout(x, p=0.5, training=self.training)
        x2 = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)

        x = self.activate_func(self.conv3(x, edge_index))
        # x = F.dropout(x, p=0.5, training=self.training)
        x3 = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)

        x = x1 + x2 + x3
        x = self.activate_func(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        return self.fc2(x)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.node_embedding.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
