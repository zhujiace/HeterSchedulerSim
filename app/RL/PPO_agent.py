import torch
from gnn import _SchedulerGNN, ValueGNN, ModelParams
import PPO_utils
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

class PPOSchedulerAgent:
    def __init__(self, model_params:ModelParams, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        self.actor = _SchedulerGNN(model_params).to(device)
        self.critic = ValueGNN(model_params).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def choose_action(self, data):
        probs = self.actor(data)
        if probs is None:
            return -1
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        graphs = Batch.from_data_list(transition_dict['graphs'])
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_graphs = Batch.from_data_list(transition_dict['next_graphs'])
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)    

        td_target = rewards + self.gamma * (1 - dones) * self.critic(next_graphs)
        td_delta = td_target - self.critic(graphs)
        advantage = PPO_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(graphs).gather(1, actions)).detach()

        actor_loss_list = []
        critic_loss_list = []
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(graphs).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(F.mse_loss(self.critic(graphs), td_target.detach()))
            actor_loss_list.append(actor_loss.item())
            critic_loss_list.append(critic_loss.item())
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        return actor_loss_list, critic_loss_list