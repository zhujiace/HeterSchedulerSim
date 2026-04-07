import torch
all_node_features = []
all_edge_index = []
x = torch.tensor([
    [1, 10, 5],
    [0, 20, 3],
    [0, 20, 3],
], dtype=torch.float)

e = torch.tensor([
    [0, 1],
    [0, 2],
], dtype=torch.long).t().contiguous()

num_nodes = 0
dag_summary_idx = x.size(0)
new_edges = torch.tensor([[dag_summary_idx] * x.size(0), torch.arange(num_nodes, num_nodes + x.size(0))], dtype=torch.long)

all_node_features.append(x)
all_node_features = torch.cat(all_node_features, dim=0)
print(all_node_features)