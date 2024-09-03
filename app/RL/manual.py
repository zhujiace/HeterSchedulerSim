from environment import SimulationEnv

env = SimulationEnv(6, 2.3)

state = env.reset()

done = False

acc_reward = 0

while not done:
    env.debug_print()
    
    decision = input(f"Please make decision for task {int(state[-7])}, seg {int(state[-6])}:")
    
    state, reward, done , info = env.step(int(decision))
    
    import os
    os.system("clear")
    
    acc_reward += reward
    print(f"Get reward {reward}, accumulated reward: {acc_reward}")

# 1 2 0 0 0
# 3 4 1 2 0
# 0
# 2 3 0
# 3 2 0
# 1 2 3 0 0
# 4 0 1
# 2 4
# 1 0 0
# 3 1 0 0
# 1 0 2 0
# 4 0 1
# 1 0
# 3 0 1
# 1 3 0
# 0
# 1 0
# 4 0
# 1 2 4
# 3 1 0
# 0
# 2 1 0
# 3 4 1 2
# 1 
# 0
# 3
# 1 2
# 0 4
# 3 1 2
# 2
# 3 1 2
# 1 3
# 1
# 4
# 1
# 3
# 1 2
