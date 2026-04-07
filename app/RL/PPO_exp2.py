import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import RLock
import pandas as pd

# 定义uti的范围
tqdm.set_lock(RLock())
# utis = [round(5.0 + i * 0.2, 1) for i in range(15)]  # [2.5, 1.6, ..., 3.9] [5.0, 5.2, ..., 7.8]
utis = [7.2, 7.4, 7.6, 7.8]
data = pd.read_csv('./random_4c4g_results_false.csv')
dict = {}
for i in range(len(data)):
    seed = data['seed'][i]
    uti = data['uti'][i]
    if uti not in dict:
        dict[uti] = []
    dict[uti].append(seed)

# 每个进程的任务函数
def run_experiment(uti, position):
    seeds = dict[uti]
    with tqdm(total=len(seeds), desc=f"UTI={uti}", unit="seed", position=position, ncols=80) as pbar:
        for seed in seeds:
            # 构建命令
            command = ["python", "PPO_train.py", "--seed", str(seed), "--uti", str(uti), "--episodes", "2000","--path", "random_4c4g_2000eps_results.csv"]
            # 运行命令
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pbar.update(1)
    return f"Completed uti={uti}"


# 并行执行
def run_parallel(utis, max_workers=2):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交初始任务
        futures = {executor.submit(run_experiment, uti, i): uti for i, uti in enumerate(utis[:max_workers])}
        remaining_utis = utis[max_workers:]
        
        # 动态提交任务
        while futures:
            # 等待任意一个任务完成
            for future in as_completed(futures):
                uti = futures.pop(future)
                try:
                    result = future.result()
                    print(result)
                except Exception as e:
                    print(f"Error for uti={uti}: {e}")
                
                # 提交下一个任务
                if remaining_utis:
                    next_uti = remaining_utis.pop(0)
                    next_position = len(futures)
                    futures[executor.submit(run_experiment, next_uti, next_position)] = next_uti
                    print(f"Submitted uti={next_uti}")

# 执行所有任务
print("开始执行所有实验")
run_parallel(utis)
print("所有实验已完成")

# # 并行执行
# def run_parallel(utis, max_workers=5):
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         # 提交任务
#         futures = {executor.submit(run_experiment, uti): uti for uti in utis}
#         # 等待任务完成
#         for future in as_completed(futures):
#             uti = futures[future]
#             try:
#                 result = future.result()
#                 print(result)
#             except Exception as e:
#                 print(f"Error for uti={uti}: {e}")

# # 分批执行
# batch_size = 5
# for i in range(0, len(utis), batch_size):
#     batch_utis = utis[i:i + batch_size]
#     print(f"Running batch: {batch_utis}")
#     run_parallel(batch_utis)
#     print(f"Completed batch: {batch_utis}")

# print("所有实验已完成")