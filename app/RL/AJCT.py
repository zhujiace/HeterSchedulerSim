import pickle
import numpy as np

from dagenv import DAGEnv


def AJCT(file, env):
    state, dep = env.reset(False)
    done = False
    time, proc_state, task_states, request = state
    task_num = len(task_states)
    task_unit = np.zeros(task_num, dtype=int)
    task_response = np.zeros(task_num, dtype=int)
    for i in range(task_num):
        for seg in task_states[i]:task_unit[i]+= seg[3]

    # print(task_unit)

    data = pickle.load(file)    
    # print(data)
    count_unit = np.zeros(task_num, dtype=int)
    count_task = np.zeros(task_num, dtype=int)
    res_task = np.zeros(task_num, dtype=int)

    print(data)
    for d in data:
        # print(d)
        timestamp = d[0]
        task_id, node_id = d[1]
        state, reward, done,_=env.step(task_id, node_id)
        # env.visualize_tasks(0)
        # input()
        time, proc_state, task_states, request = state
        
        
        if task_id == -1:
            continue
        length = task_states[task_id][node_id][3]
        count_unit[task_id] += length 
        response = timestamp + length - task_states[task_id][-1][5] * count_task[task_id]
        if response > task_response[task_id]:
            task_response[task_id] = response
        if count_unit[task_id] >= task_unit[task_id]:
            count_task[task_id] += 1
            count_unit[task_id] = 0
            res_task[task_id] += task_response[task_id]
            task_response[task_id] = 0

    # print(count_task)
    ajct = []
    for i in range(task_num):
        ajct.append(res_task[i] / count_task[i])
        # print(task_states[i][-1][-1],res_task[i] / count_task[i])
    return ajct

if __name__ == '__main__':
    env = DAGEnv(971, 3.0)
    state, dep = env.reset()
    # env.visualize_tasks(1)
    # time, proc_state, task_states, request = state
    # task_num = len(task_states)
    # task_unit = np.zeros(task_num, dtype=int)
    # for i in range(task_num):
    #     for seg in task_states[i]:task_unit[i]+= seg[3]

    # print(task_unit)

    # file = open('./Schedule_list_997_3.2/episode_50_list.pkl','rb')
    # file = open('./Schedule_list/episode_215.0_list.pkl','rb')
    # file = open('./env_trajectory/episode_165.0_list.pkl','rb')
    # file = open('./Schedule_list/126_6.0_hard/episode_22.0_list.pkl','rb')
    # file0 = open('edf_971_3.0.pkl','rb')
    # file1 = open('rm_971_3.0.pkl','rb')
    # file2 = open('sjf_971_3.0.pkl','rb')

    # ajct = AJCT(file, env)
    # print(ajct)

    # ajct0 = AJCT(file0)
    # print(ajct0)
    # ajct1 = AJCT(file1)
    # print(ajct1)
    # ajct2 = AJCT(file2)
    # print(ajct2)

    env = DAGEnv(991, 1.5)
    state, dep = env.reset()
    file = open('./Schedule_list/ViT/episode_137.0_list.pkl','rb')
    ajct = AJCT(file, env)
    print(ajct)

    # import os

    # # 指定文件夹路径
    # folder_path = './Schedule_list/ViT/'

    # # 遍历文件夹下的所有文件
    # mean0 = 100000
    # for file_name in os.listdir(folder_path):
    #     file_path = os.path.join(folder_path, file_name)
    #     if os.path.isfile(file_path):  # 确保是文件而不是文件夹
    #         # 你可以在这里处理文件，例如打开文件
    #         with open(file_path, 'rb') as file:
    #             ajct = AJCT(file, env)
    #             print(ajct, file.name)
    #             mean = sum(ajct)/len(ajct)
    #             if mean<mean0:
    #                 mean0 = mean
    #                 print("Best")

    #             # for i in range(len(ajct)):
    #             #     if ajct[i] > ajct0[i]:
    #             #         break
    #             #     print(ajct)


