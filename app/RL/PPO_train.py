import torch
from gnn import ModelParams, create_fused_graph, choose_node
from PPO_agent import PPOSchedulerAgent
import PPO_utils
from dagenv import DAGEnv
from torch_geometric.data import Data, Batch
import numpy as np
import tqdm
import atexit
from functools import partial


return_list = []
timestamp_list = []

def train_on_policy_agent(env:DAGEnv, agent:PPOSchedulerAgent, num_episodes):
    for i in range(10):
        with tqdm.tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'graphs': [], 'actions': [], 'next_graphs': [], 'rewards': [], 'dones': []}

                # seed = int(time.time())%10000
                # print("seed: ", seed)
                # env = DAGEnv(seed,3.0)
                state, dependencies = env.reset(False)
                env.client.set_simulation_timebound(1000)
                done = False
                
                timestamp, proc_state, task_states, request = state
                graphs = []
                for task_state, dependency in zip(task_states, dependencies):
                    graph = Data(x=torch.tensor(task_state, dtype=torch.float), edge_index=torch.tensor(dependency).t().contiguous())
                    # print(graph)
                    graphs.append(graph)

                fused_graph = create_fused_graph(graphs, request, device)

                while not done:
                    action = agent.choose_action(Batch.from_data_list([fused_graph]))
                    node = choose_node(fused_graph, action)
                    if node is None:
                        next_state, reward, done, _  = env.step(-1,-1)
                    else:
                        # print(f"Chosen Node for scheduling: {node}")
                        # print(f"Selection probabilities: {selection_probs}")
                        # env.visualize_tasks(node[0])
                        next_state, reward, done, _  = env.step(node[0], node[1])
                        # print(f"time: {state[0]}, reward:{reward}")
                        # env.visualize_tasks(node[0])
                        # input()
                    next_timestamp, next_proc_state, next_task_states, request = next_state
                    graphs = []
                    for task_state, dependency in zip(next_task_states, dependencies):
                        graph = Data(x=torch.tensor(task_state, dtype=torch.float), edge_index=torch.tensor(dependency).t().contiguous())
                        graphs.append(graph)
                    next_fused_graph = create_fused_graph(graphs, request, device)
                    if action != -1:
                        transition_dict['graphs'].append(fused_graph)
                        transition_dict['actions'].append(action)
                        transition_dict['next_graphs'].append(next_fused_graph)
                        transition_dict['rewards'].append(reward)
                        transition_dict['dones'].append(done)
                    fused_graph = next_fused_graph
                    episode_return += reward
                timestamp_list.append(next_timestamp)
                return_list.append(episode_return)
                actor_loss, critic_loss = agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list, timestamp_list

if __name__ == "__main__":

    atexit.register(PPO_utils.draw, return_list, timestamp_list)

    env = DAGEnv(14134,3.0)
    env.reset()
    model_params = ModelParams(
        input_dim=6, 
        embed_dim=32,
        hidden_dim=64, 
        is_sampling=False
        )
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500
    gamma = 0.98 # gamma越大越关注长期奖励
    lmbda = 0.95 # lambda=0则不累计 lambda=1完全蒙特卡洛
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent = PPOSchedulerAgent(model_params, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

    return_list, timestamp_list = train_on_policy_agent(env, agent, num_episodes)
    PPO_utils.draw(PPO_utils.moving_average(return_list,9), timestamp_list, "_ma")