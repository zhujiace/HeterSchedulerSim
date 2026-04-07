import pickle

filename = "./Schedule_list/ViTLlama/episode_224.0_list.pkl"
file = open(filename,'rb')
# data = pickle.load(file)  
# for d in data:
#     timestamp = d[0]
#     task_id, node_id = d[1]
#     print(task_id, node_id)

data = pickle.load(file)
from dagenv import DAGEnv
env = DAGEnv(14134, 0.4)
state, dep = env.reset()

state, dep = env.reset(False)
done = False
time, proc_state, task_states, request = state
task_num = len(task_states)

trace = []

from time import sleep

for d in data:
    print(f"recorded time: {d[0]}, action: {d[1]}", end=" ")
    timestamp = d[0] # time after action or before action ???

    task_id, node_id = d[1]
    proc_type = request[0]
    trace.append((time, proc_type, task_id, node_id))

    print(f"sys time: {time}->", end="")
    state, reward, done,_=env.step(task_id, node_id)
    time, proc_state, task_states, request = state
    print(f"{time}")
  
    if time > 380: sleep(0.5)
    if timestamp > 2000: break

from sys import exit
exit(0)

fout = open("./trained_trace.pkl", "wb")
pickle.dump(trace, fout)
fout.close()

