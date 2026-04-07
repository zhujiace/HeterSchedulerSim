import torch
from torch.utils.tensorboard import SummaryWriter
from gnn import ModelParams
from fused_graph import FusedGraphTemplate, create_fused_graph_from_states, choose_node
from PPO_agent import PPOSchedulerAgent
import PPO_utils
from dagenv import DAGEnv
from torch_geometric.data import Batch
import numpy as np
import tqdm
import atexit

import argparse

def inference_on_policy_agent(env:DAGEnv, agent:PPOSchedulerAgent, device):
    target = env.task_state[0][0][-1]*100
    print(f"Timebound: {target}")
    state, dependencies = env.reset()
    done = False
    
    timestamp, proc_state, task_states, request = state

    next_timestamp = timestamp
    graph_template = FusedGraphTemplate.from_task_states(task_states, dependencies, device)
    fused_graph = create_fused_graph_from_states(
        task_states, dependencies, request, timestamp, device, template=graph_template
    )

    while not done:
        action = agent.choose_action(Batch.from_data_list([fused_graph]))
        # print(f"Time: {next_timestamp}, Action: {action}")
        node = choose_node(fused_graph, action)
        if node is None:
            next_state, reward, done, _  = env.step(-1,-1)
        else:
            next_state, reward, done, _  = env.step(node[0], node[1])
        
        next_timestamp, next_proc_state, next_task_states, request = next_state

        if target <= next_timestamp:
            return True
        
        fused_graph = create_fused_graph_from_states(
            next_task_states,
            dependencies,
            request,
            next_timestamp,
            device,
            template=graph_template,
        )
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=int(991))
    parser.add_argument('--uti', type=float, default=1.5)
    parser.add_argument('--model', type=str, default="scheduler991.pth")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = DAGEnv(args.seed, args.uti)
    env.reset()

    model_params = ModelParams(
        input_dim=8, 
        embed_dim=32,
        hidden_dim=64, 
        is_sampling=False
        )
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 1000
    gamma = 0.98
    lmbda = 0.95
    entropy_coef = 0
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent = PPOSchedulerAgent(model_params, actor_lr, critic_lr, lmbda, entropy_coef, epochs, eps, gamma, device)
    agent.actor.load_state_dict(torch.load(args.model))
    print(f"Loaded model from {args.model}")
    print("Start inference...")
    success = inference_on_policy_agent(env, agent, device)
    print(f"Schedulable: {success}")
