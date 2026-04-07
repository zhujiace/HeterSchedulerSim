import heapq
from typing import Optional
import networkx as nx
import numpy as np
import math
import csv


class DAGEDFConstructur(object):

    def __init__(self, task_states: list, dep: list, smart: int = 1):

        self.task_states = task_states 
        self.dep = dep
        self.task_num = len(task_states)

        max_num_nodes = 1
        for i in range(self.task_num):
            if len(task_states[i]) > max_num_nodes:
                max_num_nodes = len(task_states[i])

        self.graphs = np.zeros(self.task_num, dtype=object)
        self.rv_graphs = np.zeros(self.task_num, dtype=object)

        # worst time needed to reach this node (exclude the node itself)
        self.critical_path = np.zeros((self.task_num, max_num_nodes), dtype=int)
        # worst case to reach the final nodes (exclude the node itself)
        self.future_work = np.zeros((self.task_num, max_num_nodes), dtype=int)
        # wort number of nodes to reach this node (exclude the node itself)
        self.critical_node = np.zeros((self.task_num, max_num_nodes), dtype=int)
        # worst number of  to reach the final nodes (exclude the node itself)
        self.future_node = np.zeros((self.task_num, max_num_nodes), dtype=int)

        self.smart_lv = smart
        self.pre_deadline = -1 * np.ones((self.task_num, max_num_nodes), dtype=float)
        # The worst case deadline should be assigned (numerous path problem)
        self.release_time = -1 * np.ones((self.task_num, max_num_nodes), dtype=int)

        self.construct_graph()

    def pre_search_ddl(self) -> np.ndarray:
        task_states = self.task_states
        for i in range(self.task_num):
            ddl: float = float(task_states[i][0][5])
            for j in range(len(task_states[i])):
                tmpddl: float = 9999
                if self.smart_lv == 0:
                    tmpddl = round(ddl / (self.critical_node[i][j] + self.future_node[i][j] + 1) * (self.critical_node[i][j] + 1), 3)
                else:
                    tmpddl = ddl * float(task_states[i][j][3]) / (self.critical_path[i][j] + self.future_work[i][j] + task_states[i][j][3])
                self.pre_deadline[i][j] = tmpddl
        return self.pre_deadline

    def construct_graph(self) -> bool:
        # construct the graph based on the init states
        # calculate the critical path and future works to do
        for i in range(self.task_num):
            self.graphs[i] = nx.DiGraph()
            self.rv_graphs[i] = nx.DiGraph()
            task_state = self.task_states[i]

            # To track the num of node in critical path
            tmp_g = nx.DiGraph()
            tmp_rv = nx.DiGraph()
            for j in range(len(task_state)):
                self.graphs[i].add_node(j, weight=task_state[j][3])
                self.rv_graphs[i].add_node(j, weight=task_state[j][3])
                tmp_g.add_node(j, weight=1)
                tmp_rv.add_node(j, weight=1)
            for edge in self.dep[i]:
                self.graphs[i].add_edge(edge[0], edge[1])
                self.rv_graphs[i].add_edge(edge[1], edge[0])
                tmp_g.add_edge(edge[0], edge[1])
                tmp_rv.add_edge(edge[1], edge[0])

            dist = self.solve_crit_path(self.graphs[i])
            for key, value in dist.items():
                self.critical_path[i][key] = value
            dist = self.solve_crit_path(self.rv_graphs[i])
            for key, value in dist.items():
                self.future_work[i][key] = value
            dist = self.solve_crit_path(tmp_g)
            for key, value in dist.items():
                self.critical_node[i][key] = value
            dist = self.solve_crit_path(tmp_rv)
            for key, value in dist.items():
                self.future_node[i][key] = value

        return True
    
    def solve_crit_path(self, g: nx.DiGraph) -> dict:
        topological_order = list(nx.topological_sort(g))
        dist = {node: 0 for node in g.nodes}
        for node in topological_order:
            for successor in g.successors(node):
                dist[successor] = max(dist[successor], dist[node] + g.nodes[node]['weight'])
        return dist 

class DAGEDFScheduler(object):
    """ EDF scheduler tailored for the DAG simulation env.

    ** smart-lv:
    - lv-0: assign internal deadline with pre-search (fair)
    - lv-1: assign internal deadline with pre-search (prop)
    - lv-2: assign internal dynamic deadline based on running time
    - lv-3: support preemption on CPU
    """

    def __init__(self, smart: int = 1, seed: int = 143, uti: float = 2.0,
                 verbose: bool = False, processor_config: Optional[dict] = None, task_count: Optional[int] = 5):
        
        from dagenv import DAGEnv
        self.env = DAGEnv(seed, uti, processor_config=processor_config, task_count=task_count)
        self.state, self.dep = self.env.reset()
        time, proc_state, task_states, request = self.state
        min_period = 1000

        self.task_num = len(task_states)
        for i in range(self.task_num):
            if task_states[i][0][5] < min_period:
                min_period = task_states[i][0][5]
        # self.env.client.set_simulation_timebound(min_period*200)
        # self.set_bound = min_period*200
        # self.env.client.set_simulation_timebound(10000)
        self.set_bound = self.env.task_state[0][0][-1]*10

        self.task_unit = np.zeros(self.task_num, dtype=int)
        for i in range(self.task_num):
            for seg in task_states[i]: self.task_unit[i]+= seg[3]

        self.queue = []
        self.graphs = np.zeros(self.task_num, dtype=object)
        self.rv_graphs = np.zeros(self.task_num, dtype=object)

        max_num_nodes = 1
        for i in range(self.task_num):
            if len(task_states[i]) > max_num_nodes:
                max_num_nodes = len(task_states[i])

        self.smart_lv = smart

        constructor = DAGEDFConstructur(task_states, self.dep, smart)
        self.pre_deadline = -1 * np.ones((self.task_num, max_num_nodes), dtype=float)
        self.pre_deadline = constructor.pre_search_ddl()
        # The worst case deadline should be assigned (numerous path problem)
        self.release_time = -1 * np.ones((self.task_num, max_num_nodes), dtype=int)

        if verbose: print("EDF Scheduler Initialized...")
        self.verbose = verbose

        self.trajectory = []

    def check_queue(self, affinity: int) -> bool:
        time, proc_state, task_states, request = self.state
        self.queue = []
        for i in range(self.task_num):
            offset = time // (task_states[i][0][5]) * (task_states[i][0][5])
            for j in range(len(task_states[i])):
                if task_states[i][j][2]==0: continue
                if task_states[i][j][4]==0: continue
                if task_states[i][j][0]!=affinity: continue
                if task_states[i][j][4]!=task_states[i][j][3]: continue
                heapq.heappush(self.queue, (self.pre_deadline[i][j] + offset, (i,j)))
        return True

    def schedule(self) -> bool: 
        # perform schedule until fail or success
        done = True
        end_time = 0
        terminate = False
        time = -1
        # while (not terminate):
        while (time < self.set_bound) and (not terminate):
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
                    # self.env.visualize_tasks(seg[0])
                    # self.env.visualize_tasks(0)
                    self.env.visualize_all_tasks()
                    print(f"Remaining queue: {self.queue}")
                    print()
                    from time import sleep
                    input()
                    from os import system
                    system('clear')
                    # sleep(1)

                # no reservation
                self.state, reward, terminate, _ = self.env.step(seg[0], seg[1])
                self.trajectory.append((time,(seg[0], seg[1])))
                if terminate and self.state[0] < self.set_bound:
                    done = False
                    end_time = self.state[0]
                    break
                    
        
            if self.verbose:
                execution = self.env.client.query_task_execution_states()
                print("Execution Progress: ", end=None)
                for i in range(self.task_num):
                    print(f"Task {i}: {execution[i]}/{self.task_unit[i]}", end=", ")
                print()

        if self.verbose: print(f"end with time {self.state[0]}")
        if done:
            return done, self.state[0]
        else:
            return done, end_time
    
    def export(self, filename):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.trajectory, f)
        return (self.trajectory)

# def find_critical_uti(seed, uti_range=(2.0, 4.0), step=0.05):
#     low, high = uti_range
#     while high - low > step:
#         mid = (low + high) / 2
#         sche = DAGEDFScheduler(seed=seed, uti=mid, smart=0, verbose=False)
#         if sche.schedule():
#             low = mid
#         else:
#             high = mid
#     return math.ceil(high * 10) / 10 

def find_critical_uti(seed):
    uti = 3.9
    while DAGEDFScheduler(seed=seed, uti=uti, smart=0, verbose=False).schedule() == False:
        uti -= 0.1
    uti += 0.1
    return seed, uti

def write_to_csv(data, filename='critical_uti_results.csv'):
    file_exists = False
    try:
        with open(filename, mode='r', newline='') as file:
            file_exists = True
    except FileNotFoundError:
        pass

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(['Seed', 'Uti'])
        
        for seed, critical_uti in data:
            writer.writerow([seed, critical_uti])

import concurrent.futures

def run_edf(uti, seeds, processor_config=None):
    results = []
    for seed in seeds:
        # 调用DAGEDFScheduler并获取调度结果
        success, time = DAGEDFScheduler(
            seed=seed,
            uti=uti,
            smart=0,
            verbose=False,
            processor_config=processor_config,
        ).schedule()
        results.append((seed, uti, success, time))
    return results

def write_to_csv_edf(results, filename):
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        for result in results:
            writer.writerow(result)

def test_edf(utis, seeds, filename = "edf_results_random.csv", processor_config=None):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["seed", "uti", "schedulable", "timetamp"])  # 写入表头

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 提交任务
        futures = {
            executor.submit(run_edf, uti, seeds, processor_config): uti
            for uti in utis
        }
        # 等待任务完成
        for future in concurrent.futures.as_completed(futures):
            uti = futures[future]
            try:
                results = future.result()
                # 将结果写入文件
                write_to_csv_edf(results, filename)
                print(f"Completed uti={uti}")
            except Exception as e:
                print(f"Error for uti={uti}: {e}")

if __name__ == "__main__":
    num_samples = 10000
    batch_size = 10000
    num_batches = num_samples // batch_size
    # results = []

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(find_critical_uti, seed) for seed in range(0,1000)]

    #     # for seed in range(num_samples):
    #     for i, future in enumerate(concurrent.futures.as_completed(futures)):
    #         seed, uti = future.result()
    #         results.append((seed, round(uti,1)))
    #         # sche = DAGEDFScheduler(seed=seed, uti= uti, smart=0, verbose=False)
    #         print(f"Seed: {seed}, Uti: {uti:.1f}")
    #     filename = f'critical_uti.csv'
    #     write_to_csv(results, filename)
    #     results = []

    sche = DAGEDFScheduler(seed=14134, uti= 5.4, smart=0, verbose=False, processor_config={0: 2, 7: 2, 8: 2}, task_count=10)
    print(sche.schedule())
    # sche.export('./Schedule_list/ViTLlama/edf_14134_0.4_.pkl')


    # # utis = [round(2.5 + i * 0.1, 1) for i in range(15)]  # [1.5, 1.6, ..., 3.9]
    # utis = [0.5]
    # # seeds = [51, 3, 33, 22, 98, 105, 70, 111, 85, 129, 156, 175, 162, 184, 219, 224, 226, 150, 225, 202, 248, 285, 247, 303, 256, 259, 307, 341, 344, 359, 366, 323, 331, 409, 380, 386, 420, 421, 405, 428, 465, 442, 487, 531, 482, 496, 536, 540, 526, 581, 584, 555, 592, 602, 575, 570, 607, 654, 618, 662, 667, 704, 678, 683, 702, 715, 687, 718, 725, 771, 776, 795, 824, 823, 827, 803, 847, 890, 866, 852, 899, 900, 939, 941, 944, 957, 53, 52, 54, 5, 6, 26, 18, 24, 39, 14, 27, 45, 20, 67]  # seed从1到100
    # seeds = [126, 490,  997, 971, 688, 331, 681, 257, 201, 534, 696, 723, 310, 116, 734, 235, 167, 39, 495, 548, 515, 164, 977, 847, 233, 457, 991, 315, 939, 445, 607, 325, 80, 800, 324, 209, 665, 321, 967, 587, 925, 498, 887, 261, 266, 831, 690, 825, 568, 520, 139, 597, 114, 357, 245, 647, 989, 311, 431, 771, 68, 202, 727, 956, 529, 276, 400, 238, 210, 877, 735, 788, 379, 615, 795, 888, 81, 193, 616, 496, 309, 226, 878, 439, 93, 964, 89, 337, 214, 476, 872, 767, 653, 643, 359, 937, 769, 850, 316, 94]
    # seeds = [126, 991, 997, 681, 14134]
    # test_edf(utis, seeds, filename = "edf_random_2c1g_ViTLlama.csv")
