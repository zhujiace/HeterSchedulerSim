# 
# Copy Right. The EHPCL Authors.
#

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    # Action-level reward: immediate feedback for scheduling or reserving.
    invalid_schedule_reward: float = 0
    valid_schedule_reward: float = 2.0
    forced_noop_reward: float = 0.0
    avoidable_noop_penalty: float = -0.1

    # Time-step shaping reward: encourages progress and utilization at each timestamp.
    time_step_penalty: float = 0
    utilization_reward_scale: float = 0.0
    on_time_job_completion_reward: float = 0.0

    # Episode-level terminal reward: encourages successful completion over failure.
    miss_deadline_reward: float = 0
    simulation_completion_reward: float = 1000

    # Optional completion-quality reward: rewards finishing earlier than the time bound.
    early_completion_bonus: bool = False
    early_completion_scale: float = 1.0


DEFAULT_REWARD_CONFIG = RewardConfig()

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
                phase_reward: bool = False,
                edf_like: bool = False,
                processor_config: Optional[dict] = None,
                task_count: int = 5) -> None:
        import sys
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sim_path = os.path.join(current_dir, '../../src/python')
        if sim_path not in sys.path:
            sys.path.append(sim_path)

        self.seed = seed
        self.client = None
        self.proc_type_limit = 10
        self.processor_names = {
            0: "CPU",
            1: "CPUBigCore",
            2: "CPULittleCore",
            3: "DataCopy",
            4: "DataCopyHTD",
            5: "DataCopyDTH",
            6: "PE",
            7: "GPU",
            8: "FPGA",
            9: "UNKNOWN",
        }
        self.default_processor_config = {0: 2, 7: 2}
        if processor_config is None:
            processor_config = self.default_processor_config
        self.processor_config = self._normalize_processor_config(processor_config)
        self.enabled_proc_types = [proc_type for proc_type, count in self.processor_config.items() if count > 0]

        from rand import MultiHardwareDAGLimitedGenerator
        self.task_generator = MultiHardwareDAGLimitedGenerator(
            self.seed,
            uti=utilization,
            n=task_count,
            hardware_specs=self._build_hardware_specs(),
        )

        self.reward_config = DEFAULT_REWARD_CONFIG
        self.execution_progress_reward_scale = 1.0

        self.phase_reward = phase_reward
        self.edf_like = edf_like
        if edf_like:
            print("Enabling EDF Like Reward... To turn off, set the parameter `False`")

        self.invalid_schedule_count = 0

    def set_early_completion_bonus(self, enabled: bool) -> None:
        self.reward_config = replace(self.reward_config, early_completion_bonus=enabled)

    def _get_processor_utilization(self) -> float:
        if not self.proc_states:
            return 0.0
        busy_count = sum(proc_state[1] != 0 for proc_state in self.proc_states)
        return busy_count / len(self.proc_states)

    def _task_completed(self, task_id: int) -> bool:
        return all(segment[4] == 0 for segment in self.task_state[task_id])

    def _has_schedulable_segment_for_current_request(self) -> bool:
        if self.request_index >= len(self.request_space):
            return False

        request_affinity = self.request_space[self.request_index][0]
        for task_state in self.task_state:
            for segment in task_state:
                if segment[0] != request_affinity:
                    continue
                if segment[1] != -1:
                    continue
                if segment[2] == 0:
                    continue
                if segment[4] == 0:
                    continue
                return True
        return False

    def _init_job_tracking(self) -> None:
        self.task_periods = np.array([task[0] for task in self.tasks], dtype=int)
        self.job_deadlines = self.task_periods.copy()
        self.job_rewarded = np.zeros(self.task_num, dtype=bool)

    def _collect_job_completion_reward(self) -> float:
        reward = 0.0
        if self.reward_config.on_time_job_completion_reward == 0:
            return reward

        for task_id in range(self.task_num):
            if (not self.job_rewarded[task_id]) and self._task_completed(task_id) and self.current_time <= self.job_deadlines[task_id]:
                reward += self.reward_config.on_time_job_completion_reward
                self.job_rewarded[task_id] = True
        return reward

    def _update_job_release_tracking(self) -> None:
        for task_id in range(self.task_num):
            period = self.task_periods[task_id]
            if period > 0 and self.current_time > 0 and self.current_time % period == 0:
                self.job_deadlines[task_id] = self.current_time + period
                self.job_rewarded[task_id] = False

    def _normalize_processor_config(self, processor_config: dict) -> dict:
        normalized = {}
        for proc_type, count in processor_config.items():
            proc_type = int(proc_type)
            count = int(count)
            if proc_type < 0 or proc_type >= self.proc_type_limit:
                raise ValueError(f"Invalid processor type id {proc_type}")
            if count < 0:
                raise ValueError(f"Processor count must be non-negative, got {count} for type {proc_type}")
            if count > 0:
                normalized[proc_type] = count
        if not normalized:
            raise ValueError("processor_config must enable at least one processor type")
        if 0 not in normalized:
            raise ValueError("processor_config must include CPU(0) because entry/exit nodes are fixed to CPU")
        return normalized

    def _build_hardware_specs(self) -> list:
        hardware_specs = []
        for proc_id in range(self.proc_type_limit):
            hardware_specs.append({
                "id": proc_id,
                "name": self.processor_names.get(proc_id, f"PROC_{proc_id}"),
                "enabled": proc_id in self.processor_config,
                "seg_min": 1,
                "seg_max": 10,
            })
        return hardware_specs

    def __del__(self):
        del self.client

    def reset(self, flash_client = True):

        if flash_client:
            del self.client
            from client import SimulatorClient
            self.client = SimulatorClient("../../build/main")

            for proc_type, proc_count in self.processor_config.items():
                self.client.create_processor(proc_type, proc_count)
            # self.client.set_simulation_timebound(200)
            # for i in range(self.task_num):
            #     if task_states[i][0][5] < min_period:
            #         min_period = task_states[i][0][5]
            # self.env.client.set_simulation_timebound(min_period*20)

            self.tasks = self.task_generator.generate_tasksets()
            for task in self.tasks:
                self.client.create_dag_task(task)
            self.task_num = len(self.tasks)
            self.task_state = np.zeros(self.task_num, dtype=tuple)
            # self.client.set_simulation_timebound(self.tasks[-1][0] + 5)
            self.time_bound = self.tasks[0][0] * 10 + 5
            self.client.set_simulation_timebound(self.time_bound)
            # print(f"Simulation Time Bound: {self.tasks[0][0]*100}")
    
            self.client.start_simulation()
        else:
            # self.task_state = np.zeros(self.task_num, dtype=tuple)
            self.reset_client()

        self.proc_num_count = np.zeros(self.proc_type_limit, dtype=int)
        for proc_type, proc_count in self.processor_config.items():
            self.proc_num_count[proc_type] = proc_count
        self.proc_busy = [0] * int(np.sum(list(self.processor_config.values())))
        self.proc_locks = np.ones(self.proc_type_limit, dtype=bool)
        for proc_type in self.enabled_proc_types:
            self.proc_locks[proc_type] = False
        self.request_index = 0
        self.request_space = []

        self.terminated = False
        self.survive_score = 0
        self.schedule_score = 0
        self.execution_score = 0
        self.invalid_schedule_count = 0
        self.current_time = self.client.get_current_time_stamp()

        self.trajectory = []
        self._init_job_tracking()

        if self.edf_like:
            # calculate the average segment length of each task
            # print("Solving segment deadlines...")
            self.query_state()
            self.query_dependency()
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
            if self._has_schedulable_segment_for_current_request():
                reward = self.reward_config.avoidable_noop_penalty
            else:
                reward = self.reward_config.forced_noop_reward
        else:
            # attempt to schedule
            reward = self.find_procId_schedule(self.request_space[self.request_index][0],taskId, segId)

        if self.edf_like and reward >= 0:
            reward += self.check_edf_like_reward(taskId, segId)
        terminate = False
        exe_reward = 0
        length = 0
        if (not self.find_next_request()):
            while (self.request_index >= len(self.request_space)) and (not terminate):
                r, terminate, l = self.update_time()
                exe_reward += r
                length += l
                if not terminate:
                    self.query_request_space()
            # while exe_reward < self.miss_deadline_reward:
            #     exe_reward -= self.miss_deadline_reward

        exe_reward *= self.execution_progress_reward_scale
        # weight = [1.0, 1.0, 1.0, 0.5, 0.5]
        # exe_reward *= weight[taskId]
        self.trajectory.append([self.current_time,(taskId, segId), exe_reward + reward])

        return self.query_state(seek_request=False), exe_reward + reward, terminate, {"release": length}

    def check_edf_like_reward(self, taskId:int, segId: int) -> int:
        task_states = self.task_state
        self.queue = []
        
        import heapq
        offset = self.current_time/(task_states[taskId][0][5]) * (task_states[taskId][0][5])
        for j in range(len(task_states[taskId])):
            if task_states[taskId][j][2]==0: continue
            if task_states[taskId][j][4]==0: continue
            if task_states[taskId][j][0]!=task_states[taskId][segId][0]: continue
            if task_states[taskId][j][4]!=task_states[taskId][j][3]: continue
            heapq.heappush(self.queue, (self.pre_ddl[taskId][j] + offset, j))
        
        heapq.heappush(self.queue, (self.pre_ddl[taskId][segId] + offset, segId))

        penalty = 0
        while self.queue!=[]:
            ddl, seg = heapq.heappop(self.queue)
            if seg!=segId:
                penalty -= task_states[taskId][seg][3]
            else: break
        return penalty

    def find_procId_schedule(self, procAffinity:int, taskId: int, segId: int) -> float:
        """Automaitcally find the <procId> of the given schedule request,
        return the invalid schedule penalty immediately if detect the 
        affinity does not match. Will further invoke the `schedule` function.
        
        """
        if (procAffinity!=self.task_state[taskId][segId][0]):
            return self.reward_config.invalid_schedule_reward
        for i in range(len(self.proc_states)):
            if (self.proc_states[i][0]==procAffinity) and (self.proc_states[i][1]==0):
                break
        if (i >= len(self.proc_states)): return self.reward_config.invalid_schedule_reward
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
            return self.reward_config.invalid_schedule_reward

        # TODO: add more reward judge rule here
        # return self.valid_schedule_reward + self.task_state[taskId][segId][3]
        return self.reward_config.valid_schedule_reward

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
        for i in self.enabled_proc_types:
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
        for i in range(self.task_num):
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

        reward = -self.reward_config.time_step_penalty
        if changenotice:
            self.prev_proc_state = self.proc_states.copy()
            self.prev_task_state = self.task_state.copy()
        
            self.execution_score += self.client.update_processor_and_task()
            self.query_state(seek_request=False)

            unlock_flag = np.zeros(self.proc_type_limit, dtype=bool)
            # condition 1, there's more processor of this type
            considered_procs = self.enabled_proc_types
            for i in considered_procs:
                prev_power = np.sum([(p[0] == i) and (p[1] == 0) for p in self.prev_proc_state])
                cur_power = np.sum([(p[0] == i) and (p[1] == 0) for p in self.proc_states])
                unlock_flag[i] |= (cur_power > prev_power)
            # condition 2, there's segment relesase (of this proc type)
            length = 0
            for i in considered_procs:
                for j in range(len(self.task_state)):
                    for k in range(len(self.task_state[j])):
                        if self.task_state[j][k][0] == i:
                            # print(self.task_state[j][k][2], self.prev_task_state[j][k][2])
                            if self.task_state[j][k][2] > self.prev_task_state[j][k][2]:
                                length += self.task_state[j][k][3]
                                # print("error")
                                # print(f"proc {i} unlock task {j} seg {k} len {self.task_state[j][k][3]}")
                                unlock_flag[i] = True; break
                    if unlock_flag[i]: break
            # condition 3, there's task release (unlock all)
            for i in considered_procs:
                if unlock_flag[i]: break
                for j in range(len(self.task_state)):
                    if self.current_time%self.task_state[j][0][5] ==0:
                        unlock_flag[i] = True
                        length += self.task_state[j][0][3]
            for i in considered_procs:
                self.proc_locks[i] &= (not unlock_flag[i])

        if self.reward_config.utilization_reward_scale != 0:
            reward += self.reward_config.utilization_reward_scale * self._get_processor_utilization()

        reward += self._collect_job_completion_reward()
        self._update_job_release_tracking()

        if self.phase_reward:
            for i in range(len(self.task_state)):
                if self.current_time%self.task_state[i][0][5]==0:
                    for j in range(len(self.task_state[i])):
                        reward += self.task_state[i][j][3]
        

        terminate = False
        if self.client.does_task_miss_deadline():
            reward += self.reward_config.miss_deadline_reward
            terminate = True
            self.trajectory = []
        elif self.client.is_simulation_completed():
            reward += self.reward_config.simulation_completion_reward
            if self.reward_config.early_completion_bonus:
                reward += max(0.0, self.time_bound - self.current_time) * self.reward_config.early_completion_scale
            self.terminated = True
            terminate = True

        return reward, terminate, length

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
            name = self.processor_names.get(state[i][0], f"P{state[i][0]}")
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

    def visualize_all_tasks(self):

        import networkx as nx
        from dagviz import visualize_dag

        # 获取所有任务的可视化字符串
        task_visualizations = []
        for task_id in range(len(self.task_state)):
            g = nx.DiGraph()
            nodes = []
            state = self.task_state[task_id]
            for i in range(len(state)):
                name = self.processor_names.get(state[i][0], f"P{state[i][0]}")
                name += f'{i}({state[i][3] - state[i][4]}/{state[i][3]})'

                # segment ready -> green; not ready -> red
                color = "\033[32m" if state[i][2]==1 else "\033[31m"
                if state[i][1]!=-1: color = "\033[33m"
                if state[i][4]==0: color = "\033[34m"
                name = color + name + "\033[0m"
                nodes.append(name)
            g.add_nodes_from(nodes)
            for edge in self.dependencies[task_id]:
                g.add_edge(nodes[edge[0]], nodes[edge[1]])

            # 获取可视化字符串
            vis_str = str(visualize_dag(g, round_angle=True))
            task_visualizations.append(vis_str.split('\n'))

        # 确定每列的最大行数
        max_lines = max(len(lines) for lines in task_visualizations)

        # 填充所有列表到相同长度
        for lines in task_visualizations:
            while len(lines) < max_lines:
                lines.append('')

        # 设置每行最多显示的任务数
        tasks_per_row = 2
        total_tasks = len(task_visualizations)

        # 计算需要多少行
        num_rows = (total_tasks + tasks_per_row - 1) // tasks_per_row

        # 分多行打印
        column_width = 60  # 每列的宽度

        for row in range(num_rows):
            # 计算当前行的任务范围
            start_idx = row * tasks_per_row
            end_idx = min((row + 1) * tasks_per_row, total_tasks)

            # 打印当前行的所有任务
            for i in range(max_lines):
                line = ""
                for j in range(start_idx, end_idx):
                    # 截断或填充每列内容以保持对齐
                    col_content = task_visualizations[j][i]
                    if len(col_content) > column_width:
                        col_content = col_content[:column_width-3] + "..."
                    else:
                        col_content = col_content.ljust(column_width)
                    line += col_content + "  "  # 列之间的间隔
                print(line)

            # 在行之间添加分隔线（可选）
            # if row < num_rows - 1:
            #    print("-" * int((column_width - 5) * tasks_per_row ))

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
        
        return int(progress*self.execution_progress_reward_scale) + timestamp

if __name__ == "__main__":
    env = DAGEnv(14134,2.0, True, True)
    env.reset()
    env.step(1,0)
    env.visualize_tasks(1)
    env.step(-1,-1)
    env.step(-1,-1)
    env.step(2,0)
    env.step(-1,-1)
    _, r, _, _ = env.step(1,2)
    print(f'reward: {r}')
    env.visualize_tasks(1)

    
