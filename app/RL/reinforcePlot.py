import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 定义空的DataFrame，指定列名为第一个字典的键
data = {'Worker': 5, 'Global Episode': 2000, 'Reward': 388.19999999999976, 'Running Reward': 383.5815364908793}
data2 = {'Survive': 4, 'Schedule': 58, 'Execution': 83.0, 'Endtime': 46, 'Invalid': 0}
data.update(data2)
df = pd.DataFrame(columns=data.keys())

# 从文件中读取日志数据
with open('sim-res-uti23.log', 'r', encoding='utf-8') as file:
    log_data = file.read()

# 解析日志文件
lines = log_data.strip().split('\n')
record = {}
for i in range(len(lines)):
    tmp : str = lines[i]
    info = {}
    if (tmp.startswith("{")):
        # data needed
        info = eval(tmp)
    else:
        continue
   
    df = pd.concat([df, pd.DataFrame([info])])

df = df.sort_values("Global Episode")
df = df[(df["Global Episode"]<=3500000)]


df['Mean Reward'] = df['Reward'].expanding().mean()

fig, ax1 = plt.subplots()

ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward', color='tab:blue')
ax1.plot(df["Global Episode"], df["Reward"], color='tab:blue', label='Reward')
ax1.tick_params(axis='y', labelcolor='tab:blue')


ax1.plot(df["Global Episode"], df["Mean Reward"], color='tab:orange', label='Mean')


fig.tight_layout()
plt.title('Episode vs Reward and Mean')
plt.show()
