import torch
import numpy as np
from gnn import SchedulerGNN, _SchedulerGNN, ModelParams

class SchedulerAgent:
    def __init__(self, model_params:ModelParams, gamma=0.99):
        self.model = _SchedulerGNN(model_params)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.gamma = gamma
        self.rewards = []  # 存储奖励
        self.actions = []  # 存储选择的动作
        self.log_probs = []  # 存储动作的log概率

    def choose_action(self, data):
        # node, action, selection_probs = self.model(data)
        # if action != -1:
        #     log_prob = torch.log(selection_probs[action])  # 获取对应动作的log概率
        #     return node, log_prob, selection_probs
        # return None, -1, []

        probs = self.model(data)
        if probs is None:
            return -1
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        # for group in batch.unique():  # 遍历每个分组
        #     mask = (batch == group)   # 当前分组掩码
        #     group_probs = probs[mask]  # 当前组的概率分布
        #     # action = torch.multinomial(group_probs.squeeze(1), num_samples=1, replacement=False)
        #     action_dist = torch.distributions.Categorical(group_probs.squeeze(1))
        #     action = action_dist.sample()
        return action.item()

    def store_transition(self, log_prob, reward):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def update_policy(self):
        R = 0
        policy_loss = []

        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            policy_loss.insert(0, -self.log_probs.pop(0) * R)

        loss = torch.stack(policy_loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.rewards = []
        self.log_probs = []

    def update_temperature(self):
        self.model.update_temperature()
