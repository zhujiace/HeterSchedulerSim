import torch
from torch.utils.tensorboard import SummaryWriter
from gnn import ModelParams
from fused_graph import FusedGraphTemplate, create_fused_graph_from_states, choose_node
from PPO_agent import PPOSchedulerAgent
from PPO_inference import inference_on_policy_agent
import PPO_utils
from dagenv import DAGEnv
from torch_geometric.data import Batch
import numpy as np
import tqdm
import atexit
from functools import partial

import argparse
import csv
import os
import pickle
from datetime import date

from AJCT import AJCT

def parse_processor_config(config_str: str) -> dict:
    processor_config = {}
    for item in config_str.split(","):
        item = item.strip()
        if not item:
            continue
        proc_type, count = item.split(":", 1)
        processor_config[int(proc_type)] = int(count)
    return processor_config

def train_on_policy_agent(env:DAGEnv, agent:PPOSchedulerAgent, num_episodes, device, return_list, timestamp_list, writer, mode):
    print(env.task_state[0][0][-1],env.task_state[1][0][-1],env.task_state[2][0][-1],env.task_state[3][0][-1],env.task_state[4][0][-1])
    target = env.task_state[0][0][-1]*10
    print(target)
    done_num = 0
    JCT_penalty = 0
    trajectory_list = []
    rollout_device = torch.device("cpu")
    for i in range(10):
        with tqdm.tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'graphs': [], 'actions': [], 'next_graphs': [], 'rewards': [], 'dones': []}

                state, dependencies = env.reset()
                done = False
                
                timestamp, proc_state, task_states, request = state
                graph_template = FusedGraphTemplate.from_task_states(
                    task_states, dependencies, rollout_device
                )

                next_timestamp = timestamp
                fused_graph = create_fused_graph_from_states(
                    task_states,
                    dependencies,
                    request,
                    timestamp,
                    rollout_device,
                    template=graph_template,
                )

                while not done:
                    action = agent.choose_action(Batch.from_data_list([fused_graph]))
                    node = choose_node(fused_graph, action)
                    # proc_states = env.check_proc_states()
                    if node is None:
                        next_state, reward, done, _  = env.step(-1,-1)
                    else:
                        next_state, reward, done, _  = env.step(node[0], node[1])
                    
                    pre_timestamp = next_timestamp
                    next_timestamp, next_proc_state, next_task_states, request = next_state
                    
                    if done:
                        next_fused_graph = fused_graph
                    else:
                        next_fused_graph = create_fused_graph_from_states(
                            next_task_states,
                            dependencies,
                            request,
                            next_timestamp,
                            rollout_device,
                            template=graph_template,
                        )
                    if action != -1:
                        transition_dict['graphs'].append(fused_graph)
                        transition_dict['actions'].append(action)
                        transition_dict['next_graphs'].append(next_fused_graph)
                        transition_dict['rewards'].append(reward)
                        transition_dict['dones'].append(done)
                    fused_graph = next_fused_graph
                    episode_return += reward
                if target <= next_timestamp:
                    if mode == "test":
                        # output_file = f"./Schedule_list/971_3.0_w/episode_{num_episodes / 10 * i + i_episode + 1}_list.pkl"
                        output_file = f"./Schedule_list/ViTLlama/uti04_episode_{num_episodes / 10 * i + i_episode + 1}_list.pkl"
                        with open(output_file, 'wb') as f:
                            pickle.dump(env.trajectory, f)
                        # AJCT(open(output_file, 'rb'), env)
                    elif mode == "default":
                        done_num += 1
                else:
                    done_num = 0

                timestamp_list.append(next_timestamp)
                return_list.append(episode_return)
                # trajectory_list.append(env.trajectory)
                # print(transition_dict['rewards'])
                actor_loss, critic_loss = agent.update(transition_dict)
                if writer is not None:
                    writer.add_scalar('train/return', episode_return, global_step=num_episodes / 10 * i + i_episode + 1)
                    writer.add_scalar('train/actor_loss', np.mean(actor_loss), global_step=num_episodes / 10 * i + i_episode + 1)
                    writer.add_scalar('train/critic_loss', np.mean(critic_loss), global_step=num_episodes / 10 * i + i_episode + 1)
                    writer.add_scalar('train/timestamp', next_timestamp, global_step=num_episodes / 10 * i + i_episode + 1)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
                if done_num >= 1 and mode == "default":
                    return return_list, timestamp_list, done_num
    return return_list, timestamp_list, done_num

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=int(14134))
    parser.add_argument('--uti', type=float, default=3.0)
    parser.add_argument('--episodes', type=int, default=int(500))
    parser.add_argument('--draw', action='store_true')
    parser.add_argument('--path', type=str, default="ppo_train_results.csv")
    parser.add_argument('--label', type=str, default="default")

    parser.add_argument("--entropy_coef", type=float, default=0, help="policy entropy")
    parser.add_argument("--mode", type=str, default="default", help="")
    parser.add_argument("--processor_config", type=str, default="0:2,7:2,8:2",
                        help="processor config in the format type:count,type:count")
    parser.add_argument("--task_count", type=int, default=5)
    parser.add_argument("--early_completion_bonus", action="store_true")

    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    writer = None
    if args.draw:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        run_name = '{}_{}_{}eps_{}t_{}'.format(
            args.seed, args.uti, args.episodes, args.task_count, args.label
        )
        log_dir = os.path.join(current_dir, 'PPO_runs', date.today().isoformat(), run_name)
        writer = SummaryWriter(log_dir=log_dir)

    return_list = []
    timestamp_list = []
    if args.draw:
        atexit.register(PPO_utils.draw_return, return_list, timestamp_list, "_" + str(args.seed) + "_" + str(args.uti))

    processor_config = parse_processor_config(args.processor_config)
    env = DAGEnv(
        args.seed,
        args.uti,
        processor_config=processor_config,
        task_count=args.task_count,
    )
    env.set_early_completion_bonus(args.early_completion_bonus)
    env.reset()
    # print(env.task_state[-1][0][-1])
    model_params = ModelParams(
        input_dim=8, # x = torch.cat([x, affinity_request.unsqueeze(1), timestamp], dim=1)
        embed_dim=32,
        hidden_dim=64, 
        is_sampling=False
        )
    actor_lr = 3e-4
    critic_lr = 1e-3
    num_episodes = args.episodes
    gamma = 0.98 # gamma越大越关注长期奖励
    lmbda = 0.95 # lambda=0则不累计 lambda=1完全蒙特卡洛
    entropy_coef = args.entropy_coef
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent = PPOSchedulerAgent(model_params, actor_lr, critic_lr, lmbda, entropy_coef, epochs, eps, gamma, device)

    return_list, timestamp_list, done_num = train_on_policy_agent(env, agent, num_episodes, device, return_list, timestamp_list, writer, args.mode)

    # np.save('mask1.npy', np.array(timestamp_list))
    # np.save("2c1g_reward_681_1.5_2.npy", np.array(return_list))

    if args.draw:
        PPO_utils.draw_return(PPO_utils.moving_average(return_list,9), timestamp_list, "_ma")

    output_file = args.path
    file_exists = os.path.isfile(output_file)

    if done_num != 0:
        result = True
    else:
        result = False

    with open(output_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["seed", "uti", "schedulable"])
        writer.writerow([args.seed, args.uti, result])
        file.flush()


if __name__ == "__main__":
    main()
