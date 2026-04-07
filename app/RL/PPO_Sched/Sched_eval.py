import torch
import numpy as np
from gnn import ModelParams, choose_node
from agent import SchedulerAgent
from dagenv import DAGEnv
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import argparse

from PPO_Sched.Sched_nn import Actor
from PPO_Sched.Sched_util import ReplayBuffer, Normalization, RewardScaling, create_fused_graph, get_s
from PPO_Sched.Sched_agent import Sched_agent

def evaluate_policy(args, env_evaluate:DAGEnv, agent:Sched_agent, state_norm, device, visible = False):
    times = 3
    evaluate_reward = 0
    timestamp = []
    for _ in range(times):
        s, edge = env_evaluate.reset(False)
        # env.client.set_simulation_timebound(args.timebound)
        done = False
        episode_reward = 0
        while not done:
            # 构建state
            # timestamp, proc_state, task_states, request = state
            # graphs = []
            # for task_state, dependency in zip(task_states, edge):
            #     graph = Data(x=torch.tensor(task_state, dtype=torch.float), edge_index=torch.tensor(dependency).t().contiguous())
            #     graphs.append(graph)
            # s = create_fused_graph(graphs, request, timestamp, device)
            s, t = get_s(s, edge, device)
            # if visible:
            #     print(request)
            if s.mask.float().sum() == s.mask.size(0):
                s_, r, done, _  = env_evaluate.step(-1,-1)
                if visible:
                    print("wait")
                    input()
            else:
                if state_norm is None and args.use_state_norm:
                    state_norm = Normalization(args, shape=s.x.shape)
                    s.x = state_norm(s.x, update=False).float()
                elif args.use_state_norm:
                    s.x = state_norm(s.x, update=False).float()
                # 选择actio
                # a = agent.evaluate(s)
                a, a_logprob = agent.choose_action(s)
                node = choose_node(s, a)
                # 执行action
                if visible:
                    print("Action: ", a)
                    print(f"Chosen Node for scheduling: {node}")
                    env_evaluate.visualize_tasks(node[0])
                    s_, r, done, _  = env_evaluate.step(node[0], node[1])
                    print(f"time: {t}, reward:{r}")
                    env_evaluate.visualize_tasks(node[0])
                    input()
                else:
                    s_, r, done, _  = env_evaluate.step(node[0], node[1])
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward
        timestamp.append(t)
    return evaluate_reward / times, timestamp

def main(args, seed, uti, device, label):
    # 创建环境
    env = DAGEnv(seed,uti)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 创建agent和回放器
    agent = Sched_agent(args)

    evaluate_reward = evaluate_policy(args, env, agent, None, device, visible=True)

    print(evaluate_reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameter Setting for Agent")
    # network
    parser.add_argument("--state_dim", type=int, default=int(7), help="The number of node features")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    # train
    parser.add_argument("--timebound", type=int, default=int(1e3), help=" Maximum number of episode steps")
    parser.add_argument("--max_train_steps", type=int, default=int(2e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()
    
    device = torch.device("cpu")
    args.device = device
    main(args, seed=0, uti=2.0, device=device, label="eval")