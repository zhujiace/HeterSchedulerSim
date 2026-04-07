import pickle
import numpy as np

from dagenv import DAGEnv


def AJCT(file, env):
    state, dep = env.reset(False)
    done = False
    time, proc_state, task_states, request = state
    task_num = len(task_states)

    ready_length = 0
    for i in range(task_num):
        for seg in task_states[i]:
            if seg[2] == 1:
                ready_length += seg[4]
    initial_ready_length = ready_length
    data = pickle.load(file)    

    r = 0
    pre_timestamp = 0
    time_list = []
    reward_list = []
    release_len = []
    for d in data:
        timestamp = d[0]
        if timestamp != pre_timestamp:
            # print(pre_timestamp, ready_length, r)
            # input()
            time_list.append(pre_timestamp) 
            reward_list.append(r)
            release_len.append(ready_length)
            r = 0   
        else:
            r += d[2]
        pre_timestamp = timestamp
        task_id, node_id = d[1]

        state, reward, done, info = env.step(task_id, node_id)
        time, proc_state, task_states, request = state
        ready_length = info["release"]
        # release_len.append(ready_length)
        # release_time.append(time)
    release_len[0] = initial_ready_length
    return time_list, reward_list, release_len

if __name__ == '__main__':
    env = DAGEnv(681, 3.0)
    state, dep = env.reset()

    import os

    file = open('./Schedule_list/681_3.0/episode_482.0_list.pkl','rb')

    time_list, reward_list, release_len = AJCT(file,env)

    # print(time_list, reward_list)

    import pandas as pd
    df = pd.DataFrame({'time': time_list, 'reward': reward_list, 'release': release_len})
    df.to_csv('./Schedule_list/681_3.0/episode_482.0_list.csv', index=False)

