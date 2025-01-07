import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os

def compute_advantage(gamma, lmbda, td_delta): # Generalized Advantage Estimation, GAE
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def draw(return_list, timestamp_list, label = ""):
        # 创建主图
    fig, ax1 = plt.subplots()
    episodes_list = list(range(len(return_list)))
    # 绘制第一个 Y 轴：Returns vs Episodes
    color = 'blue'
    ax1.plot(episodes_list, return_list, label='Returns', color=color)
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Returns', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # 创建第二个 Y 轴共享 X 轴
    ax2 = ax1.twinx()
    color = 'orange'
    ax2.plot(episodes_list, timestamp_list, label='Timestamps', color=color, linestyle='--')
    ax2.set_ylabel('Timestamps', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加标题
    plt.title('Returns and Timestamps over Episodes')

    # 保存图像
    current_time = datetime.now().strftime('%m%d%H%M')
    date_time = datetime.now().strftime('%m%d')

    folder_path = f'./Rewards/{date_time}'
    filename = f'{folder_path}/reward_{current_time}{label}.pdf'

    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(filename)

    # 显示图像
    plt.show()
    print(f"Save at {filename}")