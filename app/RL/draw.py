import matplotlib.pyplot as plt
from datetime import datetime

# 示例数据
return_list = [1, 2, 3, 4, 5]  # Replace with your actual data
timestamp_list = [0.1, 0.2, 0.3, 0.4, 0.5]  # Replace with your actual data
episodes_list = list(range(len(return_list)))

# 创建主图
fig, ax1 = plt.subplots()

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
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
plt.savefig(f'./Rewards/reward_and_timestamps_{current_time}.pdf')

# 显示图像
plt.show()
