import torch
import torch.nn as nn
import torch.nn.functional as F

# model definition 
from train import PGNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PGNetwork(94, 5).to(device)

# load model weight
model.load_checkpoint('./ckpt/sim-success-uti20.pth')

# set to evaluation model
model.eval()

from environment import SimulationEnv
env = SimulationEnv(6122)
state = env.reset()


# Start prediction
with torch.no_grad():
    done = False
    while not done:
        output = model(torch.FloatTensor(state).to(device))
        probabilities = F.softmax(output, dim=0)
        predicted_class = torch.argmax(probabilities).item()
    
        state, reward, done, info = env.step(predicted_class)
        env.client.print()
        a = input()
