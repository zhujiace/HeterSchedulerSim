# 
# Copy Right. The EHPCL Authors.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class PGAgent(object):
    def __init__(self, env, device:str = "cuda", LR = 0.00001):
        self.state_dim = len(env.query_state_lazy())
        self.action_dim = len(env.action_space)
        print(self.state_dim,self.action_dim)

        if device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.GAMMA = 0.95 # discount factor

        # init N Monte Carlo transitions in one game
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        # init network parameters
        self.network = PGNetwork(state_dim=self.state_dim, action_dim=self.action_dim)
        self.network.initialize_weights()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)

    def save_checkpoint(self, file_path):
        self.network.save_checkpoint(file_path)

    def load_checkpoint(self, file_path):
        self.network.load_state_dict(torch.load(file_path))

    def choose_action(self, observation):
        """choose action by softmax (probability) \\
        single inference, running on CPU

        Args:
            observation (tuple): state variable, 94*1
        Returns:
            actions (int): one of [0,1,2,3,4]
        """
        
        observation = torch.FloatTensor(observation)
        network_output = self.network.forward(observation)
        with torch.no_grad():
            max_output = torch.max(network_output)
            stable_output = network_output - max_output
            prob_weights: np.ndarray = F.softmax(stable_output, dim=0).cpu().numpy()

        action = np.random.choice(range(prob_weights.shape[0]),
                                  p=prob_weights)  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # run on GPU
        self.network.to(self.device)

        # Step 1: Calulate the value of each step (reverse)
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.GAMMA + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs -= np.mean(discounted_ep_rs)
        std_dev = np.std(discounted_ep_rs)
        if std_dev > 0:
            discounted_ep_rs /= std_dev
        discounted_ep_rs = torch.FloatTensor(discounted_ep_rs).to(self.device)

        # Step 2: forward
        softmax_input = self.network.forward(torch.FloatTensor(self.ep_obs).to(self.device))
        # all_act_prob = F.softmax(softmax_input, dim=0).detach().numpy()
        neg_log_prob = F.cross_entropy(input=softmax_input,
                                       target=torch.LongTensor(self.ep_as).to(self.device),
                                       reduction='none')

        # Step 3: backward
        loss = torch.mean(neg_log_prob * discounted_ep_rs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # clear up
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.network.to("cpu")
        

class Trainer:
    
    
    def __init__(self):
        self.parse_args()
        
        from environment import SimulationEnv
        self.env = SimulationEnv(self.args.s, self.args.u/10.0)
        self.env.reset()
        self.agent = PGAgent(self.env, self.args.d, self.args.l)
        if self.args.c is not None:
            self.agent.load_checkpoint(self.args.c)
        self.logfile = open(f"./logs/uti{self.args.u}_seed{self.args.s}.log", 'a+')
        self.start_ep = self.args.e

        self.record = 11
        
    def parse_args(self):
        from argparse import ArgumentParser
        parser = ArgumentParser(description="Training Agent Args")
        parser.add_argument("-s", type=int, default=61,
                            help="The random seed to use to generate tasks")
        parser.add_argument("-u", type=int, default=23,
                            help="The utilization (*10)")
        parser.add_argument("-d", type=str, default="cuda",
                            help="To use CPU only or with GPU acceleration")
        parser.add_argument("-l", type=float, default=0.000003,
                            help="The learning rate")
        parser.add_argument("-c", type=str, default=None,
                            help="ckpt path")
        parser.add_argument("-e", type=int, default=0,
                            help="starting episode")                    
        self.args = parser.parse_args()
        
    def train(self):
        
        EPISODE = 5000000
        
        from tqdm import tqdm
        for episode in tqdm(range(self.start_ep+1, EPISODE+1)):
            state = self.env.reset(False)
            
            done = False
            while not done:
                action = self.agent.choose_action(state) 
                next_state, reward, done, _ = self.env.step(action)
                self.agent.store_transition(state, action, reward) 
                state = next_state
                if done:
                    self.agent.learn() 
                    break
            
            if episode % 1000 == 0:
                total_reward = 0
                from environment import SimulationEnv
                testenv = SimulationEnv(self.args.s, self.args.u/10.0)
                state = testenv.reset()
                done = False
                while not done:
                    action = self.agent.choose_action(state)  # direct action for test
                    state, reward, done, _ = testenv.step(action)
                    total_reward += reward
                _.update({'Global Episode': episode, 'Reward': total_reward})
                print (_, file=self.logfile)
                if (total_reward > 400) or (episode % 100000 == 0):
                    self.agent.save_checkpoint(f"ckpt/uti{self.args.u}_seed{self.args.s}_{episode}.pth")
                if (_["Endtime"] > self.record):
                    _["Endtime"] = self.record
                    self.agent.save_checkpoint(f"ckpt/uti{self.args.u}_seed{self.args.s}_{episode}.pth")

if __name__ == "__main__":
    t = Trainer()
    
    t.train()
