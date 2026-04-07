import torch
import numpy as np
from torch_geometric.data import Data, Batch

class ReplayBuffer:
    def __init__(self, args):
        self.s = []
        self.a = np.zeros((args.batch_size, 1))
        self.a_logprob = np.zeros((args.batch_size, 1))
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = []
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s.append(s)
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_.append(s_)
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = self.s
        a = torch.tensor(self.a, dtype=torch.long)  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = self.s_
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a, a_logprob, r, s_, dw, done
    
    def clear(self):
        self.s = []
        self.s_ = []
        self.a.fill(0)
        self.a_logprob.fill(0)
        self.r.fill(0)
        self.dw.fill(0)
        self.done.fill(0)
        self.count = 0

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

class Normalization:
    def __init__(self, args, shape):
        self.running_ms = RunningMeanStd(shape=shape)
        self.device = args.device

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        x = x.cpu()
        if update:
            self.running_ms.update(x)
        # print(update, x.size(), self.running_ms.mean.shape, self.running_ms.std.shape)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x.to(self.device)

class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)

def create_fused_graph(graphs, request, timestamp, device):
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
    affinity_flags = all_node_features[:, 0]
    current_proc_flags = all_node_features[:, 1]
    schedulable_flags = all_node_features[:, 2]
    affinity_request = request[0]
    mask = (
            (affinity_flags != affinity_request) |  # affinity_flags 匹配 request
            (schedulable_flags != 1) |              # 可调度标志为 1
            (current_proc_flags != -1)              # 当前未被调度
        ).unsqueeze(1)  # 转为浮点数
    mask[dag_summary_index] = True
    mask[-1] = True
    extra_feature = torch.full((all_node_features.size(0), 1), affinity_request)
    all_node_features = torch.cat((extra_feature, all_node_features), dim=1)
    all_edge_index = torch.cat(all_edge_index, dim=1)

    # 创建 fused graph
    fused_data = Data(
        x=all_node_features,
        edge_index=all_edge_index
    ).to(device)

    fused_data.dag_summary_index = dag_summary_index
    fused_data.mask = mask.to(device)

    return fused_data

def choose_node(fused_data, action):
    if action == -1:
        return None
    dag_summary_index = fused_data.dag_summary_index
    if action < dag_summary_index[0]:
        return (0, action)
    dag_index = 0
    while action > dag_summary_index[dag_index]:
        dag_index += 1
    offset = dag_summary_index[dag_index - 1]
    node_index = action - offset - 1
    return (dag_index, node_index)

def get_s(state, dependencies, device):
    timestamp, proc_state, task_states, request = state
    graphs = []
    for task_state, dependency in zip(task_states, dependencies):
        graph = Data(x=torch.tensor(task_state, dtype=torch.float), edge_index=torch.tensor(dependency).t().contiguous())
        graphs.append(graph)
    s = create_fused_graph(graphs, request, timestamp, device)
    return s, timestamp