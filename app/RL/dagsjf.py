import heapq
import networkx as nx
import numpy as np
import math
import csv

class DAGSJFScheduler(object):
    """ Shortest Job First scheduler tailored for the DAG simulation env.

    """

    def __init__(self, seed: int = 143, uti: float = 2.0,
                 verbose: bool = False):
    
        from dagenv import DAGEnv
        self.env = DAGEnv(seed, uti)
        self.state, self.dep = self.env.reset()
        time, proc_state, task_states, request = self.state

        self.task_num = len(task_states)
        self.set_bound = self.env.task_state[-1][0][-1] + 2

        self.task_unit = np.zeros(self.task_num, dtype=int)
        for i in range(self.task_num):
            for seg in task_states[i]: 
                self.task_unit[i]+= seg[3]

        if verbose: print("SJF Scheduler Initialized...")
        self.verbose = verbose

        self.trajectory = []


    def check_queue(self, affinity: int) -> bool:
        time, proc_state, task_states, request = self.state
        self.queue = []

        self.exe_states = self.env.client.query_task_execution_states()
        for i in range(self.task_num):
            
            for j in range(len(task_states[i])):
                if task_states[i][j][2]==0: continue
                if task_states[i][j][4]==0: continue
                if task_states[i][j][0]!=affinity: continue
                if task_states[i][j][4]!=task_states[i][j][3]: continue
                heapq.heappush(self.queue, (task_states[i][j][3], (i,j)))
        return True

    def schedule(self) -> bool: 
        # perform schedule until fail or success
        terminate = False

        while (not terminate):
            time, proc_state, task_states, request = self.state
            if self.verbose:
                print(f"Time Stamp {time}, Request: {request}")
            if not request:
                break

            self.check_queue(request[0])
            for i in range(request[2]):
                if self.queue == []:
                    self.state, reward, terminate, _ = self.env.step(-1,-1)
                    self.trajectory.append((time,(-1, -1)))
                    if self.verbose: print(f"No ready segments, skipping...")
                    break
                ddl, seg = heapq.heappop(self.queue)
                if self.verbose:
                    print(f"Time {time}, Head of the queue: {seg}, period: {task_states[seg[0]][seg[1]][5]}, ddl: {ddl}")
                    self.env.visualize_tasks(seg[0])
                    print()
                # no reservation
                self.state, reward, terminate, _ = self.env.step(seg[0], seg[1])
                self.trajectory.append((time,(seg[0], seg[1])))
                if terminate:
                    break
        
            if self.verbose:
                execution = self.env.client.query_task_execution_states()
                print("Execution Progress: ", end=None)
                for i in range(self.task_num):
                    print(f"Task {i}: {execution[i]}/{self.task_unit[i]}", end=", ")
                print()

        if self.verbose: print(f"end with time {self.state[0]}")
        return (self.state[0] > self.set_bound)
    
    def export(self):
        import pickle
        # print(self.trajectory)
        with open('sjf_971_3.0.pkl', 'wb') as f:
            pickle.dump(self.trajectory, f)
        return (self.trajectory)
    
class DAGRMScheduler(object):
    """ Rate Monotonic scheduler tailored for the DAG simulation env.

    """

    def __init__(self, seed: int = 143, uti: float = 2.0,
                 verbose: bool = False):
    
        from dagenv import DAGEnv
        self.env = DAGEnv(seed, uti)
        self.state, self.dep = self.env.reset()
        time, proc_state, task_states, request = self.state

        self.task_num = len(task_states)
        self.set_bound = self.env.task_state[-1][0][-1] + 2

        self.task_unit = np.zeros(self.task_num, dtype=int)
        self.period = np.zeros(self.task_num, dtype=int)
        for i in range(self.task_num):
            self.period[i] = task_states[i][-1][-1]
            for seg in task_states[i]: 
                self.task_unit[i]+= seg[3]
        self.prio = np.argsort(self.period)

        if verbose: 
            print("RM Scheduler Initialized...")
            print(f"Priority: {self.prio}")
        self.verbose = verbose

        self.trajectory = []


    def check_queue(self, affinity: int) -> bool:
        time, proc_state, task_states, request = self.state
        self.queue = []

        self.exe_states = self.env.client.query_task_execution_states()
        for i in range(self.task_num):
            for j in range(len(task_states[i])):
                if task_states[i][j][2]==0: continue
                if task_states[i][j][4]==0: continue
                if task_states[i][j][0]!=affinity: continue
                if task_states[i][j][4]!=task_states[i][j][3]: continue
                heapq.heappush(self.queue, (self.prio[i], (i,j)))
        return True

    def schedule(self) -> bool: 
        # perform schedule until fail or success
        terminate = False

        while (not terminate):
            time, proc_state, task_states, request = self.state
            if self.verbose:
                print(f"Time Stamp {time}, Request: {request}")
            if not request:
                break

            self.check_queue(request[0])
            for i in range(request[2]):
                if self.queue == []:
                    self.state, reward, terminate, _ = self.env.step(-1,-1)
                    self.trajectory.append((time,(-1, -1)))
                    if self.verbose: print(f"No ready segments, skipping...")
                    break
                ddl, seg = heapq.heappop(self.queue)
                if self.verbose:
                    print(f"Time {time}, Head of the queue: {seg}, period: {task_states[seg[0]][seg[1]][5]}, ddl: {ddl}")
                    self.env.visualize_tasks(seg[0])
                    print()
                    # input()
                # no reservation
                self.state, reward, terminate, _ = self.env.step(seg[0], seg[1])
                self.trajectory.append((time,(seg[0], seg[1])))
                if terminate:
                    break
        
            if self.verbose:
                execution = self.env.client.query_task_execution_states()
                print("Execution Progress: ", end=None)
                for i in range(self.task_num):
                    print(f"Task {i}: {execution[i]}/{self.task_unit[i]}", end=", ")
                print()

        if self.verbose: print(f"end with time {self.state[0]}")
        return (self.state[0] > self.set_bound)
    
    def export(self):
        import pickle
        # print(self.trajectory)
        with open('rm_971_3.0.pkl', 'wb') as f:
            pickle.dump(self.trajectory, f)
        return (self.trajectory)

import concurrent.futures

def run(uti, seeds, type):
    results = []
    for seed in seeds:
        # 调用DAGEDFScheduler并获取调度结果
        if type == "RM":
            success = DAGRMScheduler(seed=seed, uti=uti, verbose=False).schedule()
        elif type == "SJF":
            success = DAGSJFScheduler(seed=seed, uti=uti, verbose=False).schedule()
        results.append((seed, uti, success))
    return results

def write_to_csv(results, filename):
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        for result in results:
            writer.writerow(result)

def test(utis, seeds, type, filename = "edf_results_random.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["seed", "uti", "schedulable"])  # 写入表头

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 提交任务
        futures = {executor.submit(run, uti, seeds, type): uti for uti in utis}
        # 等待任务完成
        for future in concurrent.futures.as_completed(futures):
            uti = futures[future]
            try:
                results = future.result()
                # 将结果写入文件
                write_to_csv(results, filename)
                print(f"Completed uti={uti}")
            except Exception as e:
                print(f"Error for uti={uti}: {e}")

if __name__ == "__main__":

    # sche = DAGRMScheduler(seed=997, uti= 3.0, verbose=False)
    # print(sche.schedule())
    # sche.export()

    utis = [round(2.5 + i * 0.1, 1) for i in range(15)]  # [1.5, 1.6, ..., 3.9]
    # seeds = [51, 3, 33, 22, 98, 105, 70, 111, 85, 129, 156, 175, 162, 184, 219, 224, 226, 150, 225, 202, 248, 285, 247, 303, 256, 259, 307, 341, 344, 359, 366, 323, 331, 409, 380, 386, 420, 421, 405, 428, 465, 442, 487, 531, 482, 496, 536, 540, 526, 581, 584, 555, 592, 602, 575, 570, 607, 654, 618, 662, 667, 704, 678, 683, 702, 715, 687, 718, 725, 771, 776, 795, 824, 823, 827, 803, 847, 890, 866, 852, 899, 900, 939, 941, 944, 957, 53, 52, 54, 5, 6, 26, 18, 24, 39, 14, 27, 45, 20, 67]  # seed从1到100
    seeds = [126, 490, 997, 971, 688, 331, 681, 257, 201, 534, 696, 723, 310, 116, 734, 235, 167, 39, 495, 548, 515, 164, 977, 847, 233, 457, 991, 315, 939, 445, 607, 325, 80, 800, 324, 209, 665, 321, 967, 587, 925, 498, 887, 261, 266, 831, 690, 825, 568, 520, 139, 597, 114, 357, 245, 647, 989, 311, 431, 771, 68, 202, 727, 956, 529, 276, 400, 238, 210, 877, 735, 788, 379, 615, 795, 888, 81, 193, 616, 496, 309, 226, 878, 439, 93, 964, 89, 337, 214, 476, 872, 767, 653, 643, 359, 937, 769, 850, 316, 94]
    test(utis, seeds, type = "SJF", filename = "sjf_random_2c2g10t.csv")

    
