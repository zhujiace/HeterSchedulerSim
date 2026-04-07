from PPO_agent import PPOSchedulerAgent
from dagenv import DAGEnv
from gnn import ModelParams
from fused_graph import FusedGraphTemplate, create_fused_graph_from_states, choose_node
from torch_geometric.data import Batch
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def evaluate_agent(env:DAGEnv, agent:PPOSchedulerAgent):
    state, dependencies = env.reset(False)
    episode_return = 0
    done = False
    timestamp, proc_state, task_states, request = state
    graph_template = FusedGraphTemplate.from_task_states(task_states, dependencies, device)
    fused_graph = create_fused_graph_from_states(
        task_states, dependencies, request, timestamp, device, template=graph_template
    )

    while not done:
        action = agent.choose_action(Batch.from_data_list([fused_graph]))
        node = choose_node(fused_graph, action)
        if node is None:
            next_state, reward, done, _  = env.step(-1,-1)
        else:
            next_state, reward, done, _  = env.step(node[0], node[1])
        next_timestamp, next_proc_state, next_task_states, request = next_state
        fused_graph = create_fused_graph_from_states(
            next_task_states,
            dependencies,
            request,
            next_timestamp,
            device,
            template=graph_template,
        )
        episode_return += reward
    return episode_return, next_timestamp
