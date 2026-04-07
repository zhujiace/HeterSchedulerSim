import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from gnn import ModelParams, choose_node
from agent import SchedulerAgent
from dagenv import DAGEnv
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import argparse

from PPO_Sched.Sched_nn import Actor
from PPO_Sched.Sched_util import ReplayBuffer, Normalization, RewardScaling, create_fused_graph, get_s
from PPO_Sched.Sched_agent import Sched_agent
from Sched_eval import evaluate_policy

def main(args, seed, uti, label, number):
    # 创建环境
    env = DAGEnv(seed,uti)
    env_evaluate = DAGEnv(seed,uti)
    env.reset()
    env_evaluate.reset()
    print(env.task_state[-1][0][-1])

    # env.client.set_simulation_timebound(args.timebound)
    # env_evaluate.client.set_simulation_timebound(args.timebound)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 数据
    # file_paths = ['./data/critical_uti_0.csv', './data/critical_uti_1.csv', './data/critical_uti_2.csv']
    # data = []
    # for file_path in file_paths:
    #     df = pd.read_csv(file_path)
    #     data.append(df)
    # all_data = pd.concat(data, ignore_index=True)

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    # 创建agent和回放器
    device = args.device
    agent = Sched_agent(args)
    replay_buffer = ReplayBuffer(args)

    writer = SummaryWriter(log_dir='./PPO_Sched/runs/seed_{}_uti_{}_{}_{}'.format(seed, uti, label, number))

    state_norm = None   # Trick 2:state normalization
    if args.use_reward_norm:    # Trick 3:reward normalization
        reward_norm = Normalization(args, shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    episode_num = 0
    while total_steps < args.max_train_steps:
        # 初始化环境
        # lower_bound = max(0.0, uti - 0.5)
        # upper_bound = min(4.0, uti + 0.5)
        # np.random.seed(None)
        # random_uti = np.round(np.random.uniform(lower_bound, upper_bound), 1)
        # # print("seed:", seed, " uti:", random_uti)
        # env = DAGEnv(seed, random_uti)
        # 构建state
        s, edge = env.reset(False)
        # env.client.set_simulation_timebound(args.timebound)
        s, t = get_s(s, edge, device)
        if state_norm is None and args.use_state_norm:
            state_norm = Normalization(args, shape=s.x.shape)
            s.x = state_norm(s.x).float()
        elif args.use_state_norm:
            s.x = state_norm(s.x).float()
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        train_reward = 0
        episode_num += 1
        tmp_r = 0
        done = False
        while not done:
            episode_steps += 1

            # 选择action
            if s.mask.float().sum() == s.mask.size(0):
                s_, _, done, _  = env.step(-1,-1)
                s_, t = get_s(s_, edge, device)
                # print("episode", episode_num, "step", episode_steps,": wait, reward:", r)
                if args.use_state_norm:
                    s_.x = state_norm(s_.x, update=False).float()
            else:
                a, a_logprob = agent.choose_action(s)
                node = choose_node(s, a)
                s_, r, done, _  = env.step(node[0], node[1])
                r = r / 10
                # print("episode", episode_num, "step", episode_steps,": ", node,", reward:", r)
                s_, t = get_s(s_, edge, device)

                if args.use_state_norm:
                    s_.x = state_norm(s_.x).float()
                if args.use_reward_norm:
                    r = reward_norm(r)
                elif args.use_reward_scaling:
                    r = reward_scaling(r)

                replay_buffer.store(s, a, a_logprob, r, s_, 0, done)
                
                total_steps += 1
                train_reward += r
                # print(total_steps)
            s = s_
            
            if replay_buffer.count == args.batch_size:
                policy_loss, value_loss, entropy, actor_loss = agent.update(replay_buffer, total_steps)
                writer.add_scalar('policy_loss_seed_{}_uti_{}_{}'.format(seed, uti, label), policy_loss, global_step=total_steps)
                writer.add_scalar('value_loss_seed_{}_uti_{}_{}'.format(seed, uti, label), value_loss, global_step=total_steps)
                writer.add_scalar('entropy_seed_{}_uti_{}_{}'.format(seed, uti, label), entropy, global_step=total_steps)
                writer.add_scalar('actor_loss_seed_{}_uti_{}_{}'.format(seed, uti, label), actor_loss, global_step=total_steps)
                # replay_buffer.count = 0
                replay_buffer.clear()

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward, timestamp = evaluate_policy(args, env_evaluate, agent, state_norm, device)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t timestamp:{} \t".format(evaluate_num, evaluate_reward, timestamp))
                writer.add_scalar('eval_rewards_seed_{}_uti_{}_{}'.format(seed, uti, label), evaluate_rewards[-1], global_step=total_steps)
                writer.add_scalar('eval_timestamp_seed_{}_uti_{}_{}'.format(seed, uti, label), np.average(timestamp), global_step=total_steps)
                # Save the rewards
                if evaluate_num % args.save_freq == 0:
                    np.save('./PPO_Sched/data_train/seed_{}_uti_{}_{}.npy'.format(seed, uti, label), np.array(evaluate_rewards))

        writer.add_scalar('train_rewards_seed_{}_uti_{}_{}'.format(seed, uti, label), train_reward, global_step=episode_num)
        writer.add_scalar('train_timestamp_seed_{}_uti_{}_{}'.format(seed, uti, label), t, global_step=episode_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameter Setting for Agent")
    # network
    parser.add_argument("--state_dim", type=int, default=int(7), help="The number of node features")
    parser.add_argument("--hidden_width", type=int, default=int(64), help="The number of neurons in hidden layers of the neural network")
    # train
    parser.add_argument("--timebound", type=int, default=int(5e2), help=" Maximum number of episode steps")
    parser.add_argument("--max_train_steps", type=int, default=int(2e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")  # 2048
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")   # 64
    parser.add_argument("--lr_a", type=float, default=3e-5, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-5, help="Learning rate of critic")
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

    parser.add_argument("--seed", type=int, default=int(14134), help="Seed")
    parser.add_argument("--uti", type=float, default=2.0, help="Seed")
    parser.add_argument("--label", type=str, default="", help="Label")
    parser.add_argument("--number", type=int, default=0, help="Label")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device
    # device = torch.device("cpu")
    main(args, seed=args.seed, uti=args.uti, label=args.label, number=args.number)