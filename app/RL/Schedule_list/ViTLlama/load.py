import pickle

filename = "./episode_224.0_list.pkl"
file = open(filename,'rb')
# data = pickle.load(file)  
# for d in data:
#     timestamp = d[0]
#     task_id, node_id = d[1]
#     print(task_id, node_id)

data = pickle.load(file)
import sys
sys.path.append("/home/hamster/HeterSchedulerSim/src/python/")
from dagenv import DAGEnv
env = DAGEnv(14134, 0.4)
state, dep = env.reset()

state, dep = env.reset(False)
done = False
time, proc_state, task_states, request = state
task_num = len(task_states)

data = pickle.load(file)    
trace = []

for d in data:

    timestamp = d[0]
    task_id, node_id = d[1]
    proc_id = request[0]
    state, reward, done,_=env.step(task_id, node_id)
    time, proc_state, task_states, request = state
            
    if task_id == -1: continue
    trace.append((timestamp, proc_id, task_id, node_id))

print(trace[:10])
