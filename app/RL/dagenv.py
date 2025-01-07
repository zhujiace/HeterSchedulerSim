# 
# Copy Right. The EHPCL Authors.
#

import numpy as np

class DAGEnv:
    """ RL environment for interecting with the scheduling simulation python client.

    Main APIs:
    ---
        step: choosed by the agent, will further invoke the "schedule" API;
        schedule: perform a scheduling command;
        reset: restart the client C++ backend;
        update\_time: update the timestamp by 1
    
    Notes:
        current processor and task patterns are fixed; Using same seed and utilization
    will reproduce **exact** same tasksets.
    """

    def __init__(self, seed: int, utilization: float = 2.0, 
                edf_like: bool = False) -> None:
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sim_path = os.path.join(current_dir, '../../src/python')
        if sim_path not in sys.path:
            sys.path.append(sim_path)

        self.seed = seed
        self.client = None

        from rand import DAGTaskGenerator
        self.task_generator = DAGTaskGenerator(self.seed, uti=utilization, mode='RL')
        self.task_state = np.zeros(5, dtype=tuple)
        self.proc_type_limit = 10

        # TODO: adjust the following reward
        self.invalid_schedule_reward = 0.0          # 错误的调度指令
        self.valid_schedule_reward = 0.0            # 正确的调度指令     
        self.reserve_resource_reward = 0.0          # 资源保留
        self.execution_progress_reward_scale = 1.0  # 执行1个unit
        self.miss_deadline_reward = 0.0             # 死亡
        self.simulation_completion_reward = 100.0   # 成功

        self.edf_like = edf_like
        if edf_like:
            print("Enabling EDF Like Reward... To turn off, set the parameter `False`")

        self.invalid_schedule_count = 0

    def __del__(self):
        del self.client

    def reset(self, flash_client = True):

        if flash_client:
            del self.client
            from client import SimulatorClient
            self.client = SimulatorClient("../../build/main")

            self.client.create_processor(0, 2)
            self.client.create_processor(7, 2)
            self.client.set_simulation_timebound(200)
            # self.task_num = len(task_states)
            # for i in range(self.task_num):
            #     if task_states[i][0][5] < min_period:
            #         min_period = task_states[i][0][5]
            # self.env.client.set_simulation_timebound(min_period*20)

            self.tasks = self.task_generator.generate_tasksets()
            for task in self.tasks:
                self.client.create_dag_task(task)
    
            self.client.start_simulation()
        else:
            self.reset_client()

        self.proc_num_count = np.zeros(self.proc_type_limit, dtype=int)
        self.proc_num_count[0] = 2; self.proc_num_count[7] = 2
        self.proc_busy = [0, 0, 0, 0]
        # lock this kind of procs
        self.proc_locks = np.ones(10, dtype=bool)
        self.proc_locks[0] = False; self.proc_locks[7] = False
        self.request_index = 0
        self.request_space = []

        self.terminated = False
        self.survive_score = 0
        self.schedule_score = 0
        self.execution_score = 0
        self.invalid_schedule_count = 0
        self.current_time = self.client.get_current_time_stamp()

        if self.edf_like:
            # calculate the average segment length of each task
            self.query_state()
            self.segment_ave_len = np.zeros(len(self.task_state), dtype=float)
            for i in range(len(self.task_state)):
                tot_len: int = 0
                for j in range(len(self.task_state[i])):
                    tot_len += self.task_state[i][j][3]
                self.segment_ave_len[i] = tot_len / len(self.task_state[i])
            
            self.query_dependency()
            print("Solving segment deadlines...")
            from dagedf import DAGEDFConstructur
            constructor = DAGEDFConstructur(self.task_state, self.dependencies, 0)
            self.pre_ddl = constructor.pre_search_ddl()
        return self.query_state(), self.query_dependency()
    
    def reset_client(self) -> bool:
        return self.client.reset_client()

    def step(self, taskId: int, segId: int) -> 'tuple[tuple, float, bool, dict]':
        # TODO:
        reward = 0

        if (taskId < 0):
            # the agent give up this round (reserve some resources)
            # lock the resources
            self.proc_locks[self.request_space[self.request_index][0]] = True
            reward = self.reserve_resource_reward
        else:
            # attempt to schedule
            reward = self.find_procId_schedule(self.request_space[self.request_index][0],taskId, segId)

        exe_reward = reward
        if self.edf_like:
            exe_reward += self.check_edf_like_reward(taskId, segId)
        terminate = False
        if (not self.find_next_request()):
            while (self.request_index >= len(self.request_space)):
                r, terminate = self.update_time()
                exe_reward += r
                if terminate: break
                self.query_request_space()

        exe_reward *= self.execution_progress_reward_scale
        return self.query_state(seek_request=False), exe_reward, terminate, {}

    def check_edf_like_reward(self, taskId:int, segId: int) -> float:
        pass

    def find_procId_schedule(self, procAffinity:int, taskId: int, segId: int) -> float:
        """Automaitcally find the <procId> of the given schedule request,
        return the invalid schedule penalty immediately if detect the 
        affinity does not match. Will further invoke the `schedule` function.
        
        """
        if (procAffinity!=self.task_state[taskId][segId][0]):
            return self.invalid_schedule_reward
        for i in range(len(self.proc_states)):
            if (self.proc_states[i][0]==procAffinity) and (self.proc_states[i][1]==0):
                break
        if (i >= len(self.proc_states)): return self.invalid_schedule_reward
        return self.schedule(i, taskId, segId)

    def schedule(self, procId:int, taskId: int, segId: int) -> float:
        """Perform the schedule command.

        Args:
            procId(int): the id of processor
            taskId(int): the id of the task
            segId(int): the id of segment(node) in the task

        Returns:
            reward (float): 0 if schedule a task, -1000 if wrong behavior

        Notes:
            erromatic operations are ingored automatically, a nagative reward
        will be returned
        """

        res = self.client.schedule_segment_on_processor(procId, taskId, segId)
        if res.find("Error")!=-1 :
            self.invalid_schedule_count += 1 
            return self.invalid_schedule_reward

        # TODO: add more reward judge rule here
        return self.valid_schedule_reward

    def check_proc_states(self) -> list:
        """check the busy states of processors
        0 -> idle, 1 -> busy.
        
        Example:
        ---
        [0, 1, 0, 1] -> CPU1 busy, GPU1 busy
        """
        self.proc_states: list = list(self.client.query_processor_states())
        
        for i in range(len(self.proc_states)):
            self.proc_busy[i] = self.proc_states[i][1]
        return self.proc_busy
    
    def query_request_state_lazy(self) -> list:
        """check which processor will send requests

        returns:
        ---
        [processorAffinity, the current proc index, number of procs];
        e.g. [0, 0, 2] -> 2 CPUs are requesting, currently the first request
        """
        return self.request_space[self.request_index]

    def query_request_space(self) -> list:
        idle_counts = np.zeros(self.proc_type_limit, dtype=int)
        for proc in self.proc_states:
            if proc[1]!=0: continue
            if self.proc_locks[proc[0]]==False:
                idle_counts[proc[0]] += 1
        self.request_space = []
        for i in range(self.proc_type_limit):
            for j in range(idle_counts[i]):
                self.request_space.append([i, j, idle_counts[i]])
        self.request_index = 0
        return self.request_space

    def find_next_request(self) -> bool:
        self.request_index += 1
        while (self.request_index < len(self.request_space)) and \
              (self.proc_locks[self.request_space[self.request_index][0]]==True):
            self.request_index += 1
        return (self.request_index < len(self.request_space))

    def query_request(self):
        if self.request_index >= len(self.request_space):
            return []
        else:
            return self.request_space[self.request_index]

    def skip_busy_days(self) -> bool:
        """skip those timestamps that no processors are available
        
        """
        while not self.is_terminated():
            self.query_state()
            all_proc_busy = True
            for proc in self.proc_states:
                if (proc[1]==0) & (self.proc_locks[proc[0]]==False):
                    all_proc_busy = False; break
            if all_proc_busy:
                self.update_time()
        
        return True

    def query_state(self, seek_request = True) -> 'list':
        """return the simulator, processor, task states, requests
        
        Returns:
            [time, proc state, task states, current request];
            proc state = [p1, p2, p3, ...];
            p1 = [procType, processorState, taskIndex, segIndex];
            task states = [t1, t2, t3, ...];
            t1 = [[s1], [s2], [s3], ..., [sn]];
            s1 = affinity, currentProcessor, isSegmentReady, length, remainLength, period;
            request = [processorAffinity, index of current request, total num request of this affinity]
        """
        self.current_time = self.client.get_current_time_stamp()
        # the proc states are tuple of tuple
        self.proc_states: list = list(self.client.query_processor_states())
        # the task state has repeat the "period" multiple times to 
        # keep the format same
        for i in range(5):
            tmp = self.client.query_task_state(i)
            result = []
            for j in range(len(tmp[1])):
                result.append(list(tmp[1][j])+[tmp[0]])
            self.task_state[i] = result
        
        if seek_request: self.query_request_space()
        return [self.current_time, self.proc_states, self.task_state, \
                self.query_request()]
    
    def query_state_lazy(self) -> 'list':
        """Will **not** activately call the querys from the C++ backend,
        just read from the local storage. Consider to use this function to
        speed up your learning system.
        
        """
        return [self.current_time, self.proc_states, self.task_state, \
                self.query_request()]
    
    def is_terminated(self) -> bool:
        if self.client.does_task_miss_deadline(): return True
        if self.client.is_simulation_completed(): return True
        return False

    def update_time(self, changenotice = True) -> 'tuple[float, bool]':
        """advance the simulator by 1 time

        (***NEW Feature***) compare the task and processor state 
        with previous timestamp
        
        Returns:
            reward (float):  0 if miss ddl, 10 if complete
            terminate (bool): true if (either miss ddl / complete)
        """

        if changenotice:
            self.prev_proc_state = self.proc_states
            self.prev_task_state = self.task_state
        
            reward = self.client.update_processor_and_task()
            self.execution_score += reward
            self.query_state(seek_request=False)

            unlock_flag = np.zeros(10, dtype=bool)
            # condition 1, there's more processor of this type
            considered_procs = [0,7]
            for i in considered_procs:
                prev_power = np.sum([(p[1]==0) for p in self.prev_proc_state])
                cur_power = np.sum([(p[1]==0) for p in self.proc_states])
                unlock_flag[i] |= (cur_power > prev_power)
            # condition 2, there's segment relesase (of this proc type)
            for i in considered_procs:
                for j in range(len(self.task_state)):
                    for k in range(len(self.task_state[j])):
                        if self.task_state[j][k][0] == i:
                            if self.task_state[j][k][1] > self.prev_task_state[j][k][1]:
                                unlock_flag[i] = True; break
                    if unlock_flag[i]: break
            # condition 3, there's task release (unlock all)
            for i in considered_procs:
                if unlock_flag[i]: break
                for j in range(len(self.task_state)):
                    if self.current_time%self.task_state[j][0][5] ==0:
                        unlock_flag[i] = True
            for i in considered_procs:
                self.proc_locks[i] &= (not unlock_flag[i])

        terminate = False
        if self.client.does_task_miss_deadline():
            reward = self.miss_deadline_reward
            terminate = True
        elif self.client.is_simulation_completed():
            # reward += self.simulation_completion_reward
            reward += self.current_time
            terminate = True

        return reward, terminate

    def query_dependency(self) -> 'list':
        self.dependencies = []
        # The tasks are already stored in self.tasks
        for task in self.tasks:
            num_nodes = task[1]
            num_edges = task[2]
            # edges start from tasks[3+2*nodes] -> tasks[-1]
            tmp = task[3+2*num_nodes:]
            edges = [[tmp[i], tmp[i+1]] for i in range(0, len(tmp), 2)]
            self.dependencies.append(edges)
        return self.dependencies

    def debug_print(self):
        """Call the `printSimulatorState` method 
        """
        self.client.print()

    def visualize_tasks(self, taskId: int) -> None:
        """Require: py-dagviz
        """

        import networkx as nx
        from dagviz import visualize_dag

        g = nx.DiGraph()
        nodes = []
        state = self.task_state[taskId]
        for i in range(len(state)):
            name = 'C' if state[i][0]==0 else 'G'
            name += f'{i}({state[i][3] - state[i][4]}/{state[i][3]})'
            
            # segment ready -> green; not ready -> red
            color = "\033[32m" if state[i][2]==1 else "\033[31m"
            if state[i][1]!=-1: color = "\033[33m"
            if state[i][4]==0: color = "\033[34m"
            name = color + name + "\033[0m"
            nodes.append(name)
        g.add_nodes_from(nodes)
        for edge in self.dependencies[taskId]:
            g.add_edge(nodes[edge[0]], nodes[edge[1]])
        print(visualize_dag(g, round_angle=True))
    
    def query_maximum_reward(self, timestamp: int = None) -> int:
        """Infer the maximum reward the agent may or/not achieve.
        Note: The scheduling of RT DAG is a NP-Hard problem, therefore
        its impossible to deduce an optimal scheduling strategy and thus
        impossible to derive the **real** upper bound. The maximum reward
        here just provide a theorectical upper bound that the agent may or
        may not achieve.
        """
        if not timestamp:
            timestamp = self.current_time
        
        progress: int = 0
        for i in range(len(self.task_state)):
            tot_len: int = 0
            for j in range(len(self.task_state[i])):
                tot_len += self.task_state[i][j][3]
            progress += (timestamp//self.task_state[i][0][5])*tot_len
            progress += max(timestamp%self.task_state[i][0][5], tot_len)
        
        return int(progress*self.execution_progress_reward_scale) + self.simulation_completion_reward

if __name__ == "__main__":
    env = DAGEnv(14134,1.3)
    env.reset()
    env.visualize_tasks(1)
    env.step(1,0)
    env.visualize_tasks(1)
    env.step(1,0)
    env.visualize_tasks(1)

    