from environment import SimulationEnv

from numpy import random

random.seed(630)
env = SimulationEnv(630)

env.reset()

accumulated_reward = 0

for _ in range(1000):
    actions = env.action_space()
    if actions:
        action = actions[random.randint(0, len(actions))]
        state, reward, terminated, info = env.step(*action)
    else:
        state, reward, terminated, info = env.step()
    
    accumulated_reward += reward
    if terminated : break
    if accumulated_reward < 0: break
    
    env.debug_print()
    print(f"reward: {reward}, accumulated reward: {accumulated_reward}")
    from time import sleep
    sleep(0.1)

print(f"reward: {accumulated_reward}")
