import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import RLock

# 定义uti的范围
# utis = [round(2.5 + i * 0.1, 1) for i in range(15)]  # [2.5, 1.6, ..., 3.9] [5.0, 5.2, ..., 7.8]
# low
# seeds = [51, 3, 33, 22, 98, 105, 70, 111, 85, 129, 156, 175, 162, 184, 219, 224, 226, 150, 225, 202, 248, 285, 247, 303, 256, 259, 307, 341, 344, 359, 366, 323, 331, 409, 380, 386, 420, 421, 405, 428, 465, 442, 487, 531, 482, 496, 536, 540, 526, 581, 584, 555, 592, 602, 575, 570, 607, 654, 618, 662, 667, 704, 678, 683, 702, 715, 687, 718, 725, 771, 776, 795, 824, 823, 827, 803, 847, 890, 866, 852, 899, 900, 939, 941, 944, 957, 53, 52, 54, 5, 6, 26, 18, 24, 39, 14, 27, 45, 20, 67]  # seed从1到100
# random
seeds = [126, 490, 997, 971, 688, 331, 681, 257, 201, 534, 696, 723, 310, 116, 734, 235, 167, 39, 495, 548, 515, 164, 977, 847, 233, 457, 991, 315, 939, 445, 607, 325, 80, 800, 324, 209, 665, 321, 967, 587, 925, 498, 887, 261, 266, 831, 690, 825, 568, 520, 139, 597, 114, 357, 245, 647, 989, 311, 431, 771, 68, 202, 727, 956, 529, 276, 400, 238, 210, 877, 735, 788, 379, 615, 795, 888, 81, 193, 616, 496, 309, 226, 878, 439, 93, 964, 89, 337, 214, 476, 872, 767, 653, 643, 359, 937, 769, 850, 316, 94]
# seeds = [126]
utis = [6.0]
# seeds = seeds[:25]
# utis = [7.2, 7.4, 7.6, 7.8]
tqdm.set_lock(RLock())

# 每个进程的任务函数
def run_experiment(uti, position):
    with tqdm(total=len(seeds), desc=f"UTI={uti}", unit="seed", position=position, ncols=80) as pbar:
        for seed in seeds:
            # 构建命令
            command = ["python", "PPO_train.py", "--seed", str(seed), "--uti", str(uti), "--episodes", "500", "--path", "random_2c2g4f_results_500eps.csv", "--processor_config", "0:2,7:2,8:4", "--task_count", "10", "--label", "2c2g4f"]
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