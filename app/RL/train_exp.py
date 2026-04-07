"""
@ Author: Peter Xiao
@ Date: 2020.7.20
@ Filename: PG.py
@ Brief: 使用 蒙特卡洛策略梯度Reinforce训练CartPole-v0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
from collections import deque

# Hyper Parameters for PG Networki"i
GAMMA = 0.95  # discount factor
LR = 0.000001  # learning rate

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.backends.cudnn.enabled = False  # 非确定性算法

class ResNetBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResNetBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.relu = nn.ReLU()

        if input_dim != output_dim:
            self.residual_connection = nn.Linear(input_dim, output_dim)
        else:
            self.residual_connection = None

    def forward(self, x):
        residual = x
        out = self.relu(self.fc1(x))
        out = self.fc2(out)

        if self.residual_connection is not None:
            residual = self.residual_connection(x)

        out += residual
        out = self.relu(out)
        return out

class PGNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PGNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.resblock = ResNetBlock(64, 64)
        self.fc2 = nn.Linear(64, 48)
        self.fc3 = nn.Linear(48, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.resblock(out)
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.fc5(out)
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
    def __init__(self, env):  # 初始化
        # 状态空间和动作空间的维度
        self.state_dim = len(env.query_state_lazy())
        self.action_dim = len(env.action_space)
        print(self.state_dim,self.action_dim)

        # init N Monte Carlo transitions in one game
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        # init network parameters
        self.network = PGNetwork(state_dim=self.state_dim, action_dim=self.action_dim)
        self.network.initialize_weights()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        # self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        #self.optimizer = torch.optim.SGD(self.network.parameters(), lr=LR)


    def save_checkpoint(self, file_path):
        self.network.save_checkpoint(file_path)

    def load_checkpoint(self, file_path):
        self.network.load_state_dict(torch.load(file_path))

    def choose_action(self, observation):
        observation = torch.FloatTensor(observation)
        # observation = torch.FloatTensor(observation).to(device)
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

    def learn(self):

        self.network.to(device)

        # Step 1: 计算每一步的状态价值
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        # 注意这里是从后往前算的，所以式子还不太一样。算出每一步的状态价值
        # 前面的价值的计算可以利用后面的价值作为中间结果，简化计算；从前往后也可以
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * GAMMA + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # print(discounted_ep_rs)
        discounted_ep_rs -= np.mean(discounted_ep_rs)  # 减均值
        # discounted_ep_rs /= np.std(discounted_ep_rs)  # 除以标准差
        std_dev = np.std(discounted_ep_rs)
        if std_dev > 0:
            discounted_ep_rs /= std_dev
        discounted_ep_rs = torch.FloatTensor(discounted_ep_rs).to(device)

        # print(self.ep_obs,self.ep_as,discounted_ep_rs)
        # Step 2: 前向传播
        softmax_input = self.network.forward(torch.FloatTensor(self.ep_obs).to(device))
        # all_act_prob = F.softmax(softmax_input, dim=0).detach().numpy()
        neg_log_prob = F.cross_entropy(input=softmax_input, target=torch.LongTensor(self.ep_as).to(device), reduction='none')

        # Step 3: 反向传播
        loss = torch.mean(neg_log_prob * discounted_ep_rs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 每次学习完后清空数组
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.network.to("cpu")
    

# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 6000000  # Episode limitation
STEP = 1000  # Step limitation in an episode
TEST = 1  # The number of experiment test every 100 episode

from environment import SimulationEnv
import time


def main():
    # initialize OpenAI Gym env and dqn agent
    from dual import DualCPUEnv
    # env = DualCPUEnv(6122, 2.5)
    env = SimulationEnv(15, 2.3)
    env.reset()

    agent = PG(env)
    agent.load_checkpoint("./ckpt/sim-res-uti23_3400000.pth")
    
    endrecord = 11

    from tqdm import tqdm
    for episode in tqdm(range(3400001, EPISODE+1)):
        # initialize task

        state = env.reset(False)
        # Train
        done = False
        while not done:
            action = agent.choose_action(state)  # softmax概率选择action
            next_state, reward, done, _ = env.step(action)
        
            agent.store_transition(state, action, reward)   # 新函数 存取这个transition
            state = next_state
            if done:
                # print("stick for ",step, " steps")
                agent.learn()   # 更新策略网络
                break


        if episode % 1000 == 0:
            total_reward = 0
            for i in range(TEST):
                # testenv = SimulationEnv(int(time.time())*73%13)
                testenv = SimulationEnv(15, 2.3)
                state = testenv.reset()
                for j in range(STEP):
                    action = agent.choose_action(state)  # direct action for test
                    state, reward, done, _ = testenv.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            _.update({'Global Episode': episode, 'Reward': ave_reward})
            print (_)
            if (total_reward > 700) or (_["Endtime"]>endrecord) or (episode % 100000 == 0):
                agent.save_checkpoint(f"ckpt/sim-res-uti23_{episode}.pth")
                endrecord = max(_["Endtime"], endrecord)


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('The total time is ', time_end - time_start)
