import torch
from gnn import ModelParams
from fused_graph import FusedGraphTemplate, create_fused_graph_from_states, choose_node
from agent import SchedulerAgent
from dagenv import DAGEnv
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

def eval_scheduler(model_params:ModelParams):
    agent = SchedulerAgent(model_params)

    env = DAGEnv(14134,2.0)

    state, dependencies = env.reset()
    timestamp, proc_state, task_states, request = state
    graph_template = FusedGraphTemplate.from_task_states(task_states, dependencies, device)
    env.client.set_simulation_timebound(20000)
    while(1):
        timestamp, proc_state, task_states, request = state
        print(request)
        fused_graph = create_fused_graph_from_states(
            task_states, dependencies, request, timestamp, device, template=graph_template
        )
        # print(fused_graph)
        # print(fused_graph.dag_summary_index)

        # print(request)
        action = agent.choose_action(Batch.from_data_list([fused_graph])) # Batch.from_data_list([fused_graph])
        node = choose_node(fused_graph, action)
        print("Action: ", action)
        if node is None:
            state, reward, terminal, _  = env.step(-1,-1)
        else:
            print(f"Chosen Node for scheduling: {node}")
            # print(f"Selection probabilities: {selection_probs}")
            env.visualize_tasks(node[0])
            state, reward, terminal, _  = env.step(node[0], node[1])
            print(f"time: {state[0]}, reward:{reward}")
            env.visualize_tasks(node[0])
            input()
        if terminal: break
        # print(request)

if __name__ == "__main__":

    model_params = ModelParams(
        input_dim=8, 
        embed_dim=32,
        hidden_dim=64, 
        is_sampling=False
        )
    
    device = torch.device("cpu")
    eval_scheduler(model_params)
