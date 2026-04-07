import torch
from gnn import _SchedulerGNN, ValueGNN, ModelParams
import PPO_utils
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch.distributions import Categorical

class PPOSchedulerAgent:
    def __init__(self, model_params:ModelParams, actor_lr, critic_lr, lmbda, entropy_coef, epochs, eps, gamma, device):
        self.actor = _SchedulerGNN(model_params).to(device)
        self.critic = ValueGNN(model_params).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_coef = entropy_coef
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def choose_action(self, data):
        data = data.to(self.device)
        with torch.no_grad():
            probs = self.actor(data)
            if probs is None:
                return -1
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        if not transition_dict['graphs']:
            return [], []

        graphs = Batch.from_data_list(transition_dict['graphs']).to(self.device)
        next_graphs = Batch.from_data_list(transition_dict['next_graphs']).to(self.device)
        actions = torch.as_tensor(transition_dict['actions'], device=self.device, dtype=torch.long).view(-1, 1)
        rewards = torch.as_tensor(transition_dict['rewards'], device=self.device, dtype=torch.float32).view(-1, 1)
        dones = torch.as_tensor(transition_dict['dones'], device=self.device, dtype=torch.float32).view(-1, 1)

        with torch.no_grad():
            next_values = self.critic(next_graphs)
            old_values = self.critic(graphs)
            td_target = rewards + self.gamma * (1 - dones) * next_values
            td_delta = td_target - old_values
            advantage = PPO_utils.compute_advantage(self.gamma, self.lmbda, td_delta)
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
            old_probs = self.actor(graphs)
            old_log_probs = torch.log(old_probs.gather(1, actions).clamp_min(1e-8))

        actor_loss_list = []
        critic_loss_list = []
        for _ in range(self.epochs):
            probs = self.actor(graphs)
            values = self.critic(graphs)
            entropy = Categorical(probs).entropy().view(-1, 1)
            log_probs = torch.log(probs.gather(1, actions).clamp_min(1e-8))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            
            policy_loss = -torch.min(surr1, surr2)  # PPO损失函数
            actor_loss = torch.mean(policy_loss - self.entropy_coef * entropy)
            critic_loss = F.mse_loss(values, td_target)
            total_loss = actor_loss + critic_loss
            
            actor_loss_list.append(actor_loss.item())
            critic_loss_list.append(critic_loss.item())
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        return actor_loss_list, critic_loss_list
