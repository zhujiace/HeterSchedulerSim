import torch
import torch.nn as nn
import torch.nn.functional as F

# model definition 
from train import PGNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PGNetwork(94, 5).to(device)

# load model weight
model.load_checkpoint('./ckpt/uti23_seed496_800000.pth')

# set to evaluation model
model.eval()

from environment import SimulationEnv
env = SimulationEnv(496, 2.3)
state = env.reset()


def choose_action(observation):
    output = model(torch.FloatTensor(state).to(device))
    with torch.no_grad():
        max_output = torch.max(output)
        stable_output = output - max_output
        prob_weights = F.softmax(stable_output, dim=0).cpu().numpy()
    
    import numpy as np
    action = np.random.choice(range(prob_weights.shape[0]),
                              p=prob_weights)  # select action w.r.t the actions prob
    return action

# Start prediction
with torch.no_grad():
    done = False
    while not done:
        #output = model(torch.FloatTensor(state).to(device))
        #probabilities = F.softmax(output, dim=0)
        #predicted_class = torch.argmax(probabilities).item()
        action = choose_action(state)
    
        env.client.print()
        print(f"Schedule {env.to_schedule}, take action: {action}")
        state, reward, done, info = env.step(action)

        # a = input()
