# Copy Right
# 2024 The EHPCL Authors
# 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import time
from collections import deque
import torch.multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized
from multiprocessing import Queue
from torch.optim import Optimizer
from environment import SimulationEnv
from dual import DualCPUEnv

# Hyper Parameters for PG Network
GAMMA = 0.95  # discount factor
LR = 0.00005  # learning rate

# Use GPU
device = torch.device("cuda:0")
# torch.backends.cudnn.enabled = False  # 非确定性算法


class PGNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PGNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, action_dim)


    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        # out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                nn.init.constant_(m.bias.data, 0.01)
            # m.bias.data.zero_()

    def save_checkpoint(self, file_path):
        torch.save(self.state_dict(), file_path)
        print(f"Checkpoint saved to {file_path}")

    def load_checkpoint(self, file_path):
        self.load_state_dict(torch.load(file_path))
        print(f"Checkpoint loaded from {file_path}")


class PG(object):
    # dqn Agent
    def __init__(self, env: SimulationEnv, shared_network, optimizer):

        self.state_dim = 86
        self.action_dim = 3

        # init N Monte Carlo transitions in one game
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        # init network parameters
        # self.network = PGNetwork(state_dim=self.state_dim, action_dim=self.action_dim).to(device)
        # self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        self.network: PGNetwork = shared_network
        self.optimizer: Optimizer = optimizer
        self.lr = 0.00005

        # init some parameters
        self.time_step = 0

    def choose_action(self, observation):
        observation = torch.FloatTensor(observation).to(device)
        network_output = self.network.forward(observation)
        with torch.no_grad():
            max_output = torch.max(network_output)
            stable_output = network_output - max_output
            prob_weights = F.softmax(stable_output, dim=0).cpu().numpy()
        # prob_weights = F.softmax(network_output, dim=0).detach().numpy()

        action = np.random.choice(range(prob_weights.shape[0]),
                                  p=prob_weights)  # select action w.r.t the actions prob
        return action

    # 将状态，动作，奖励这一个transition保存到三个列表中

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def adjust_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 2.0

    def learn(self):
        self.time_step += 1

        if (self.time_step %10000 == 0):
            self.adjust_lr()

        # Step 1: Calulate the step value
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * GAMMA + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs -= np.mean(discounted_ep_rs)  # 减均值
        std_dev = np.std(discounted_ep_rs)
        if std_dev > 0:
            discounted_ep_rs /= std_dev
        discounted_ep_rs = torch.FloatTensor(discounted_ep_rs).to(device)

        # Step 2: Forward
        softmax_input = self.network.forward(torch.FloatTensor(self.ep_obs).to(device))
        # all_act_prob = F.softmax(softmax_input, dim=0).detach().numpy()
        neg_log_prob = F.cross_entropy(input=softmax_input, target=torch.LongTensor(self.ep_as).to(device), reduction='none')

        # Step 3: Backward
        loss = torch.mean(neg_log_prob * discounted_ep_rs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Step 4: Clearup
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

    def save_checkpoint(self, file_path):
        self.network.save_checkpoint(file_path)

    def load_checkpoint(self, file_path):
        self.network.load_state_dict(torch.load(file_path))

def worker(worker_id:int, env_seed:int, shared_network: PGNetwork, optimizer: Optimizer,
           global_ep: Synchronized, global_ep_r: Synchronized, res_queue: Queue):

    from environment import SimulationEnv
    env = DualCPUEnv(env_seed)
    env.reset()
    agent = PG(env, shared_network, optimizer)
    
    while global_ep.value < EPISODE:
        state = env.reset(flash_client=False)
        ep_r = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward)
            state = next_state
            ep_r += reward
            if done:
                agent.learn()
                with global_ep.get_lock():
                    global_ep.value += 1
                with global_ep_r.get_lock():
                    if global_ep_r.value == 0:
                        global_ep_r.value = ep_r
                    else:
                        global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
                res_queue.put(global_ep_r.value)
                if global_ep.value%50000==10000:
                    shared_network.save_checkpoint(f"ckpt/checkpoint{int(time.time())}_{global_ep.value}.pth")
                if global_ep.value%1000==0:
                    _.update({'NetworkTime': agent.time_step , 'Worker': worker_id, 'Global Episode': global_ep.value, 'Reward': ep_r, 'Running Reward': global_ep_r.value})
                    print(_)
                    from sys import stderr
                    print(_, file=stderr)
                break

# ---------------------------------------------------------
# Hyper Parameters
EPISODE = 3000000  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode


def main():

    state_dim = 86
    action_dim = 3

    global_network = PGNetwork(state_dim, action_dim).to(device)
    global_network.share_memory()
    optimizer = torch.optim.Adam(global_network.parameters(), lr=LR)

    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    workers = [mp.Process(target=worker, args=(i, 6122, global_network, optimizer,
                                               global_ep, global_ep_r, res_queue)) for i in range(8)]

    [w.start() for w in workers]
    [w.join() for w in workers]

if __name__ == '__main__':
    time_start = time.time()
    torch.multiprocessing.set_start_method('spawn')
    device = torch.device("cuda:0")
    main()
    time_end = time.time()
    print('The total time is ', time_end - time_start)
