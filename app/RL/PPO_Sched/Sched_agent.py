import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from PPO_Sched.Sched_nn import Actor,Critic

from torch_geometric.data import Data, Batch

class Sched_agent:
    def __init__(self, args):
        self.device = args.device
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        self.actor = Actor(args).to(self.device)
        self.critic = Critic(args).to(self.device)
        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    # def evaluate(self, s):  # When evaluating the policy, we select the action with the highest probability
    #     # s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
    #     a_prob = self.actor(s).detach().numpy().flatten()
    #     a = np.argmax(a_prob)
    #     return a

    def choose_action(self, s):
        # s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            dist = Categorical(probs=self.actor(s))
            a = dist.sample()
            a_logprob = dist.log_prob(a)
        return a.cpu().numpy(), a_logprob.cpu().numpy()
    
    def edf_action(self, s):
        with torch.no_grad():
            dist = Categorical(probs=self.actor(s))
            a = dist.sample() # 
            a_logprob = dist.log_prob(a)
        return a.numpy(), a_logprob.numpy()

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0

        batch_s = Batch.from_data_list(s).to(self.device)
        batch_s_ = Batch.from_data_list(s_).to(self.device)
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(batch_s).cpu()
            vs_ = self.critic(batch_s_).cpu()
            deltas = r + self.gamma * (1.0 - done) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        actor_loss_list = []
        policy_loss_list = []
        value_loss_list = []
        entropy_list = []

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = Categorical(probs=self.actor( Batch.from_data_list( [s[i] for i in index] ) ).cpu())
                dist_entropy = dist_now.entropy().view(-1, 1)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index].squeeze()).view(-1, 1)  # shape(mini_batch_size X 1)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_now - a_logprob[index])  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                policy_loss = -torch.min(surr1, surr2)
                actor_loss = policy_loss - self.entropy_coef * dist_entropy  # shape(mini_batch_size X 1)
                # Update actor
                # actor_loss = actor_loss.to(self.device)
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic( Batch.from_data_list( [s[i] for i in index] )).cpu()
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                # critic_loss = critic_loss.to(self.device)
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

                policy_loss_list.append(torch.mean(policy_loss).detach().item())
                value_loss_list.append(torch.mean(critic_loss).detach().item())
                entropy_list.append(torch.mean(dist_entropy).detach().item())
                actor_loss_list.append(torch.mean(actor_loss).detach().item())

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

        return sum(policy_loss_list)/len(policy_loss_list), sum(value_loss_list)/len(value_loss_list), sum(entropy_list)/len(entropy_list), sum(actor_loss_list)/len(actor_loss_list)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def update2(self, transition_dict, episode):
        s = Batch.from_data_list(transition_dict['states'])
        a = torch.tensor(transition_dict['actions'], dtype=torch.long)
        a_logprob = torch.tensor(transition_dict['logprob'], dtype=torch.float)
        r = torch.tensor(transition_dict['rewards'], dtype=torch.float)
        s_ = Batch.from_data_list(transition_dict['next_states'])
        done = torch.tensor(transition_dict['dones'], dtype=torch.float)    

        adv = []
        gae = 0
        with torch.no_grad():
            vs = self.critic(s).cpu()
            vs_ = self.critic(s_).cpu()
            deltas = r + self.gamma * (1.0 - done) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        policy_loss_list = []
        entropy_list = [] 
        actor_loss_list = []
        value_loss_list = []

        for _ in range(self.K_epochs):
            dist_now = Categorical(probs=self.actor(s).cpu())
            dist_entropy = dist_now.entropy().view(-1, 1)
            a_logprob_now = dist_now.log_prob(a.squeeze()).view(-1, 1)
            # log_probs = torch.log(self.actor(s).gather(1, a))
            ratios = torch.exp(a_logprob_now - a_logprob)

            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv  # 截断
            policy_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            actor_loss = policy_loss - self.entropy_coef * dist_entropy

            self.optimizer_actor.zero_grad()
            actor_loss.mean().backward()
            if self.use_grad_clip:  # Trick 7: Gradient clip
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer_actor.step()
            
            v_s = self.critic(s).cpu()
            critic_loss = F.mse_loss(v_target, v_s)
            # critic_loss = torch.mean(F.mse_loss(self.critic(s), td_target.detach()))

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            if self.use_grad_clip:  # Trick 7: Gradient clip
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer_critic.step()

            policy_loss_list.append(policy_loss.item())
            entropy_list.append(dist_entropy.mean().item())
            actor_loss_list.append(actor_loss.mean().item())
            value_loss_list.append(critic_loss.item())

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(episode)

        return sum(policy_loss_list)/len(policy_loss_list), sum(value_loss_list)/len(value_loss_list), sum(entropy_list)/len(entropy_list), sum(actor_loss)/len(actor_loss)