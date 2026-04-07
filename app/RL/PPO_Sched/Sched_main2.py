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
from app.RL.Sched_eval import evaluate_policy
import tqdm

def main(args, seed, uti, label, number):
    env = DAGEnv(seed,uti)
    env_evaluate = DAGEnv(seed,uti)
    env.reset()
    env_evaluate.reset()
    env.client.set_simulation_timebound(args.timebound)
    env_evaluate.client.set_simulation_timebound(args.timebound)
    np.random.seed(seed)
    torch.manual_seed(seed)

    evaluate_num = 0        # Record the number of evaluations
    return_list = []            # Record the rewards during the evaluating
    total_steps = 0         # Record the total steps during the training

    device = args.device
    agent = Sched_agent(args)

    writer = SummaryWriter(log_dir='./PPO_Sched/runs/seed_{}_{}_{}'.format(seed, label, number))

    state_norm = None   # Trick 2:state normalization
    if args.use_reward_norm:    # Trick 3:reward normalization
        reward_norm = Normalization(args, shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    for i in range(10):
        with tqdm.tqdm(total=int(args.num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(args.num_episodes / 10)):

                s, edge = env.reset()
                env.client.set_simulation_timebound(args.timebound)
                s, t = get_s(s, edge, device)
                if state_norm is None and args.use_state_norm:
                    state_norm = Normalization(args, shape=s.x.shape)
                    s.x = state_norm(s.x).float()
                elif args.use_state_norm:
                    s.x = state_norm(s.x).float()
                if args.use_reward_scaling:
                    reward_scaling.reset()

                train_reward = 0
                tmp_r = 0
                transition_dict = {'states': [], 'actions': [], 'logprob':[], 'next_states': [], 'rewards': [], 'dones': []}
                done = False

                while not done:

                    if s.mask.float().sum() == s.mask.size(0):
                        s_, r, done, _  = env.step(-1,-1)
                        tmp_r += r
                        s_, t = get_s(s_, edge, device)
                        if args.use_state_norm:
                            s_.x = state_norm(s_.x, update=False).float()
                    else:
                        a, a_logprob = agent.choose_action(s)
                        node = choose_node(s, a)
                        s_, r, done, _  = env.step(node[0], node[1])
                        s_, t = get_s(s_, edge, device)
                        r += tmp_r

                        if args.use_state_norm:
                            s_.x = state_norm(s_.x).float()
                        if args.use_reward_norm:
                            r = reward_norm(r)
                        elif args.use_reward_scaling:
                            r = reward_scaling(r)

                        transition_dict['states'].append(s)
                        transition_dict['actions'].append(a.item())
                        transition_dict['logprob'].append(a_logprob.item())
                        transition_dict['next_states'].append(s_)
                        transition_dict['rewards'].append(r)
                        transition_dict['dones'].append(done)
                        tmp_r = 0
                        
                        total_steps += 1
                        train_reward += r
                    s = s_
                return_list.append(train_reward)
                episode = args.num_episodes / 10 * i + i_episode + 1
                policy_loss, value_loss, entropy, actor_loss = agent.update2(transition_dict, episode)
                writer.add_scalar('episode_rewards_seed_{}_{}'.format(seed, label), train_reward, global_step=episode)
                writer.add_scalar('timestamp_seed_{}_{}'.format(seed, label), t, global_step=episode)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (episode),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
                writer.add_scalar('policy_loss_seed_{}_{}'.format(seed, label), policy_loss, global_step=episode)
                writer.add_scalar('value_loss_seed_{}_{}'.format(seed, label), value_loss, global_step=episode)
                writer.add_scalar('entropy_seed_{}_{}'.format(seed, label), entropy, global_step=episode)
                writer.add_scalar('actor_loss_seed_{}_{}'.format(seed, label), actor_loss, global_step=episode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameter Setting for Agent")
    # network
    parser.add_argument("--state_dim", type=int, default=int(7), help="The number of node features")
    parser.add_argument("--hidden_width", type=int, default=int(64), help="The number of neurons in hidden layers of the neural network")
    # train
    parser.add_argument("--timebound", type=int, default=int(5e2), help=" Maximum number of episode steps")
    parser.add_argument("--num_episodes", type=int, default=int(1e3), help=" Maximum number of training steps")
    parser.add_argument("--max_train_steps", type=int, default=int(1e3), help="")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")  # 2048
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")   # 64
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

    parser.add_argument("--label", type=str, default="", help="Label")
    parser.add_argument("--number", type=int, default=0, help="Label")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.device = device
    # device = torch.device("cpu")
    main(args, seed=14134, uti=2.0, label=args.label, number=args.number)