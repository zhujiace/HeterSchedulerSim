import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class TaskSchedulerGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, is_sampling:bool, temperature=1.0, decay_rate=0.99):
        super(TaskSchedulerGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Multi-layer perceptron (MLP) for better feature extraction
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 3, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

        self.is_sampling = is_sampling
        self.temperature = temperature
        self.decay_rate = decay_rate

    def forward(self, data):
        # Get the input data
        x, edge_index = data.x, data.edge_index
        dag_summary_index = data.dag_summary_index

        schedule_flags = data.x[:, -1]
        dag_idx = 0
        node_index_offset = 0

        # Graph Convolutional layers for feature propagation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Generate scheduling scores for all nodes (only for schedulable nodes)
        node_scores = []
        node_candidates = []
        for i in range(x.size(0)-1):  # Iterate over all nodes
            if schedule_flags[i] == 1 and i not in dag_summary_index:
                node_embedding = x[i]
                if i >= dag_summary_index[dag_idx]:
                    node_index_offset = dag_summary_index[dag_idx] + 1
                    dag_idx += 1
                dag_emb = x[dag_summary_index[dag_idx]]
                global_emb = x[-1]
                combined_input = torch.cat([node_embedding, dag_emb, global_emb], dim=0)  # Concatenate embeddings

                # Pass through MLP to better extract features
                score = self.mlp(combined_input)  # Use MLP to generate a better score for this node
                node_scores.append(score)
                node_candidates.append([dag_idx, i-node_index_offset, score])
                print(f"dag{dag_idx}, node{i-node_index_offset}, score: {score}")

        if len(node_scores) > 0:
            # Stack the scores for all nodes
            node_scores = torch.stack(node_scores, dim=0).squeeze(-1)  # Shape (num_nodes,)

            # Apply softmax to the scores for all schedulable nodes
            selection_probs = F.softmax(node_scores/self.temperature, dim=0)
            if self.is_sampling:
                selected_node = torch.multinomial(selection_probs, 1)
                return node_candidates[selected_node]
            else:
                selected_node = torch.argmax(selection_probs)
                return node_candidates[selected_node]
        else:
            return -1

    def update_temperature(self):
        self.temperature *= self.decay_rate


def create_fused_graph(graphs):
    """
    Create a fused graph by adding dag_summary nodes and a global_summary node.
    """
    all_node_features = []
    all_edge_index = []
    num_nodes = 0  # Track the total number of nodes
    dag_summary_nodes = []  # Store DAG summary nodes
    dag_summary_index = []

    # Process each graph
    for graph in graphs:
        x, edge_index = graph.x, graph.edge_index

        # 记录当前图的节点数
        current_num_nodes = num_nodes

        # 更新节点总数
        num_nodes += x.size(0)

        # 偏移当前图的边索引
        offset_edge_index = edge_index + current_num_nodes

        # 更新全局节点特征和边索引
        all_node_features.append(x)
        all_edge_index.append(offset_edge_index)

        # 生成 DAG summary 节点
        dag_summary_node = torch.mean(x, dim=0, keepdim=True)  # 使用所有节点特征的均值作为 DAG summary
        dag_summary_nodes.append(dag_summary_node)
        all_node_features.append(dag_summary_node)

        # 将 DAG summary 节点连接到 DAG 中的每个节点
        dag_summary_idx = num_nodes  # DAG summary 节点的索引
        new_edges = torch.tensor([[dag_summary_idx] * x.size(0), torch.arange(current_num_nodes, num_nodes)], dtype=torch.long)
        all_edge_index.append(new_edges)
        dag_summary_index.append(dag_summary_idx)

        # 将 DAG summary 节点的特征添加到 all_node_features
        # all_node_features.append(dag_summary_node)
        num_nodes += 1  # 更新节点总数，增加一个 DAG summary 节点

    # 生成 global summary 节点，使用所有 DAG summary 节点的特征均值
    global_summary = torch.mean(torch.cat(dag_summary_nodes, dim=0), dim=0, keepdim=True)
    all_node_features.append(global_summary)

    global_summary_idx = num_nodes  # Global summary 节点的索引
    num_nodes += 1  # 更新节点总数，增加 global summary 节点

    # 连接每个 DAG summary 节点到 global summary 节点
    for i in range(len(dag_summary_nodes)):
        dag_summary_idx = num_nodes - len(dag_summary_nodes) - 1 + i  # 获取 DAG summary 的索引
        all_edge_index.append(torch.tensor([[global_summary_idx], [dag_summary_idx]], dtype=torch.long))

    # 拼接所有的节点特征和边索引
    all_node_features = torch.cat(all_node_features, dim=0)
    all_edge_index = torch.cat(all_edge_index, dim=1)

    # 创建 fused graph
    fused_data = Data(
        x=all_node_features,
        edge_index=all_edge_index
    )

    # 将 DAG summary 和 global summary 信息添加为图的附加属性
    # fused_data.dag_summary = torch.cat(dag_summary_nodes, dim=0)  # 合并所有 DAG summary 节点
    fused_data.dag_summary_index = dag_summary_index
    # fused_data.global_summary = global_summary  # Global summary 节点

    return fused_data


# Example Usage

# 定义两个 DAG 图
node_features1 = torch.tensor([
    [1, 10, 5, 1],  # Node 1
    [0, 20, 3, 0],  # Node 2
], dtype=torch.float)

edge_index1 = torch.tensor([
    [0, 1],  # Edge from Node 1 -> Node 2
], dtype=torch.long).t().contiguous()

node_features2 = torch.tensor([
    [1, 15, 7, 2],  # Node 1
    [0, 25, 4, 2],  # Node 2
    [1, 10, 2, 1],  # Node 3
], dtype=torch.float)

edge_index2 = torch.tensor([
    [0, 1],  # Edge from Node 1 -> Node 2
    [1, 2],  # Edge from Node 2 -> Node 3
], dtype=torch.long).t().contiguous()

# 创建 Data 对象
graph1 = Data(x=node_features1, edge_index=edge_index1)
graph2 = Data(x=node_features2, edge_index=edge_index2)
print(graph1)

# 生成 fused graph
fused_data = create_fused_graph([graph1, graph2])
print(fused_data)
print(fused_data.x)
print(fused_data.edge_index)

# # 定义 GNN 模型
input_dim = node_features1.size(1)
hidden_dim = 64

model = TaskSchedulerGNN(input_dim, hidden_dim, is_sampling=True)

# 在 fused graph 上执行前向传播
selection_node = model(fused_data)

print(f"Selection probabilities: {selection_node}")
