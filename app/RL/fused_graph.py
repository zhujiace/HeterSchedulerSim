import torch
from torch_geometric.data import Data


def _to_bidirectional_edges(edge_index):
    reverse_edge_index = edge_index.flip(0)
    return torch.cat([edge_index, reverse_edge_index], dim=1)


def _dependency_to_edge_index(dependency, device):
    if len(dependency) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.as_tensor(dependency, dtype=torch.long, device=device).t().contiguous()


class FusedGraphTemplate:
    def __init__(self, node_counts, dependencies, device, bidirectional=True):
        self.device = torch.device(device)
        self.bidirectional = bidirectional
        self.node_counts = [int(count) for count in node_counts]
        self.num_tasks = len(self.node_counts)
        self._edge_index_cache = {}

        self.task_node_slices = []
        self.dag_summary_index = []

        total_nodes = 0
        edge_parts = []
        for node_count, dependency in zip(self.node_counts, dependencies):
            start = total_nodes
            end = start + node_count
            self.task_node_slices.append((start, end))
            total_nodes = end

            edge_index = _dependency_to_edge_index(dependency, self.device)
            if edge_index.numel() > 0:
                edge_index = edge_index + start
                if self.bidirectional:
                    edge_index = _to_bidirectional_edges(edge_index)
                edge_parts.append(edge_index)

            summary_idx = total_nodes
            self.dag_summary_index.append(summary_idx)

            summary_edges = torch.stack(
                [
                    torch.full((node_count,), summary_idx, dtype=torch.long, device=self.device),
                    torch.arange(start, end, dtype=torch.long, device=self.device),
                ],
                dim=0,
            )
            if self.bidirectional:
                summary_edges = _to_bidirectional_edges(summary_edges)
            edge_parts.append(summary_edges)
            total_nodes += 1

        self.global_summary_idx = total_nodes
        total_nodes += 1
        self.total_nodes = total_nodes

        global_edges = []
        for summary_idx in self.dag_summary_index:
            edge_index = torch.tensor(
                [[self.global_summary_idx], [summary_idx]],
                dtype=torch.long,
                device=self.device,
            )
            if self.bidirectional:
                edge_index = _to_bidirectional_edges(edge_index)
            global_edges.append(edge_index)

        if global_edges:
            edge_parts.extend(global_edges)

        self.edge_index = (
            torch.cat(edge_parts, dim=1)
            if edge_parts
            else torch.empty((2, 0), dtype=torch.long, device=self.device)
        )
        self._edge_index_cache[self.device] = self.edge_index

    @classmethod
    def from_graphs(cls, graphs, device, bidirectional=True):
        node_counts = [graph.x.size(0) for graph in graphs]
        dependencies = []
        for graph in graphs:
            if graph.edge_index.numel() == 0:
                dependencies.append([])
            else:
                dependencies.append(graph.edge_index.t().cpu().tolist())
        return cls(node_counts, dependencies, device, bidirectional=bidirectional)

    @classmethod
    def from_task_states(cls, task_states, dependencies, device, bidirectional=True):
        node_counts = [len(task_state) for task_state in task_states]
        return cls(node_counts, dependencies, device, bidirectional=bidirectional)

    def edge_index_for(self, device):
        device = torch.device(device)
        cached = self._edge_index_cache.get(device)
        if cached is None:
            cached = self.edge_index.to(device)
            self._edge_index_cache[device] = cached
        return cached

    def materialize(self, task_states, request, timestamp, device=None):
        data_device = self.device if device is None else torch.device(device)
        feature_parts = []
        summary_parts = []

        for task_state in task_states:
            task_x = torch.as_tensor(task_state, dtype=torch.float32, device=data_device)
            feature_parts.append(task_x)
            task_summary = task_x.mean(dim=0, keepdim=True)
            summary_parts.append(task_summary)
            feature_parts.append(task_summary)

        global_summary = torch.mean(torch.cat(summary_parts, dim=0), dim=0, keepdim=True)
        feature_parts.append(global_summary)

        x = torch.cat(feature_parts, dim=0)

        request_tensor = torch.as_tensor(request, dtype=torch.float32, device=data_device).view(1, -1)
        timestamp_tensor = torch.as_tensor(timestamp, dtype=torch.float32, device=data_device).view(1, 1)

        affinity_flags = x[:, 0]
        current_proc_flags = x[:, 1]
        schedulable_flags = x[:, 2]
        affinity_request = request_tensor[0, 0]
        mask = (
            (affinity_flags != affinity_request)
            | (schedulable_flags != 1)
            | (current_proc_flags != -1)
        ).unsqueeze(1)
        mask[self.dag_summary_index] = True
        mask[self.global_summary_idx] = False

        fused_data = Data(
            x=x,
            edge_index=self.edge_index_for(data_device),
        )
        fused_data.dag_summary_index = self.dag_summary_index
        fused_data.global_summary_idx = self.global_summary_idx
        fused_data.mask = mask
        fused_data.request = request_tensor
        fused_data.timestamp = timestamp_tensor
        fused_data.bidirectional = self.bidirectional
        return fused_data


def create_fused_graph(graphs, request, timestamp, device, bidirectional=True, template=None):
    if template is None:
        template = FusedGraphTemplate.from_graphs(graphs, device, bidirectional=bidirectional)
    task_states = [graph.x for graph in graphs]
    return template.materialize(task_states, request, timestamp, device=device)


def create_fused_graph_from_states(task_states, dependencies, request, timestamp, device, bidirectional=True, template=None):
    if template is None:
        template = FusedGraphTemplate.from_task_states(
            task_states, dependencies, device, bidirectional=bidirectional
        )
    return template.materialize(task_states, request, timestamp, device=device)


def choose_node(fused_data, action):
    if action == -1:
        return None
    if action == fused_data.global_summary_idx:
        return None
    if action in fused_data.dag_summary_index:
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


def _build_mock_dag(node_features, edges):
    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
    )


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    graphs = [
        _build_mock_dag(
            node_features=[
                [0, -1, 1, 3, 3, 10, 0, 1],
                [0, -1, 1, 2, 2, 10, 0, 0],
                [0, 0, 0, 1, 1, 10, 1, 0],
            ],
            edges=[
                [0, 0],
                [0, 1],
                [1, 2],
            ],
        ),
        _build_mock_dag(
            node_features=[
                [1, -1, 1, 5, 5, 20, 0, 1],
                [1, -1, 0, 4, 4, 20, 1, 0],
                [1, -1, 1, 2, 2, 20, 0, 0],
                [1, 1, 1, 1, 1, 20, 1, 1],
            ],
            edges=[
                [0, 0],
                [0, 1],
                [1, 2],
                [1, 3],
            ],
        ),
        _build_mock_dag(
            node_features=[
                [0, -1, 1, 6, 6, 15, 0, 1],
                [0, -1, 1, 3, 1, 15, 1, 0],
            ],
            edges=[
                [0, 1],
            ],
        ),
    ]

    request = [0, 1, 3]
    timestamp = [[12.0]]

    print("=== Mock DAG Summary ===")
    for idx, graph in enumerate(graphs):
        print(f"DAG {idx}: nodes={graph.x.size(0)}, edges={graph.edge_index.size(1)}")

    for bidirectional in [False, True]:
        fused_graph = create_fused_graph(graphs, request, timestamp, device, bidirectional=bidirectional)

        print(f"\n=== Fused Graph (bidirectional={bidirectional}) ===")
        print(f"device: {fused_graph.x.device}")
        print(f"x shape: {tuple(fused_graph.x.shape)}")
        print(f"edge_index shape: {tuple(fused_graph.edge_index.shape)}")
        print(f"dag_summary_index: {fused_graph.dag_summary_index}")
        print(f"global_summary_idx: {fused_graph.global_summary_idx}")
        print(f"request: {fused_graph.request.tolist()}")
        print(f"timestamp: {fused_graph.timestamp.tolist()}")

        print("\nnode features:")
        print(fused_graph.x.cpu())

        print("\nedge_index:")
        print(fused_graph.edge_index.cpu())

        print("\nmask:")
        print(fused_graph.mask.squeeze(1).cpu())

        print("\n=== Action Mapping Check ===")
        sample_actions = [0, 1, fused_graph.dag_summary_index[0], fused_graph.global_summary_idx]
        for action in sample_actions:
            print(f"action {action} -> {choose_node(fused_graph, action)}")


if __name__ == "__main__":
    main()
