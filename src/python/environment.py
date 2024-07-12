# 
# Copy Right. The EHPCL Authors.
#

import numpy as np
from client import SimulatorClient
from rand import TaskRandomGenerator

class SimulationEnv:
    """ RL environment for interecting with the scheduling simulation python client.

    Main APIs:
    ---
        action_space: return tuple of available processor to schedule \n
        schedule: perform a scheduling command \n
        reset: restart the client \n
        update_time: update the timestamp by 1

    Notes:
        current processor and task patterns are fixed; Using same seed and utilization
    will reproduce **exact** same tasksets.
    """

    def __init__(self, seed: int, utilization: float = 2.0) -> None:
        self.seed = seed
        self.client = None
        self.utilization = utilization

        self.task_generator = TaskRandomGenerator(self.seed)
        self.accumulated_reward = 0

        self.task_state = np.zeros(5, dtype=tuple)
        self.avail_schedules = []
        self.searched_at = -1

        self.self_suspension = True
        self.to_schedule = [-1, -1]

    
    def __del__(self):
        del self.client

    def reset(self):
        del self.client
        self.client = SimulatorClient("../../build/main")

        self.client.create_processor(0, 2)
        self.client.create_processor(7, 2)

        tasks = self.task_generator.generate(self.utilization)
        for task in tasks:
            self.client.create_heter_ss_task(task[0], 2, (0,7), task[1])
        
        self.client.start_simulation()
        self.searched_at = -1
        self.to_schedule = [-1, -1]
        self.schedule_space()
        self.find_next_task()
        self.terminated = False

    def action_space(self) -> list:
        if self.terminated:
            return []
        return [-1, 0, 1, 2, 3]

    def schedule_space(self) -> list:
        """search and return list of available schedulings

        Returns:
            list[tuple]: (i,j,k), schedule the k-th segment of j-th task on i-th processor
        """
        if self.client is None:
            from sys import stderr
            print("Please use reset() first before find the action spaces", file=stderr)
        self.query_state()
        if self.searched_at == self.client.get_current_time_stamp():
            return self.avail_schedules

        self.avail_schedules = []
        for i, proc_state in enumerate(self.proc_states):
            # ignore non-preemptive procs
            if proc_state[1] > 1: continue
            for j, task_state in enumerate(self.task_state):
                if self.self_suspension:
                    if task_state[2] < 999999: continue
                    # already completed, negative index
                    if task_state[1] < 0: continue
                    # not same affinity
                    # if proc_state[0] != task_state[4][task_state[1]][0]: continue
                    self.avail_schedules.append((i, j, task_state[1]))
                else:
                    for k,segment in enumerate(task_state[1]):
                        # not the same affinity
                        if segment[0]!=proc_state[0]: continue
                        # already on some processor, useless to schedule on others
                        if segment[1]<999999: continue
                        # segment not ready yet
                        if segment[2]==0: continue
                        # segment is already completed
                        if segment[4]==0: continue
                        self.avail_schedules.append((i, j, k))

        self.searched_at = self.client.get_current_time_stamp()

        return self.avail_schedules

    def find_next_task(self) -> None:
        # update self.to_schedule
        flag = False
        for taskId in range(self.to_schedule[0]+1, 5):
            if flag: break
            for action in self.avail_schedules:
                if action[1] == taskId:
                    self.to_schedule[0] = action[1]
                    self.to_schedule[1] = action[2]
                    flag = True
                    break
        if not flag:
            self.to_schedule[0] = -1

    def step(self, procId: int) -> 'tuple[tuple, int, bool, dict]':
        reward = 0
        if (procId >= 0):
            reward = self.schedule(procId, self.to_schedule[0], self.to_schedule[1])
            if reward < 0:
                return self.query_state(), reward, True, {"valid": False}
        self.find_next_task()
        while (self.to_schedule[0] == -1):
            # no action
            rewardacc, terminate = self.update_time()
            reward = reward + rewardacc
            if terminate:
                self.terminated = True
                return self.query_state(), reward, True, {}
            self.schedule_space()
            self.find_next_task()

        return self.query_state(), reward, False, {}

    def schedule(self, procId:int, taskId: int, segId: int) -> int:
        """Perform the schedule command. The avail action is updated.

        Returns:
            reward (int): 0 if schedule a task, -1000 if wrong behavior
        """
        if (not (procId, taskId, segId) in self.avail_schedules):
            return -1000
        res = self.client.schedule_segment_on_processor(procId, taskId, segId)
        if res.find("Error")!=-1 :
            return -1000

        new_avail_schedules = []
        for action in self.avail_schedules:
            if (action[0]==procId): continue
            if (action[1]==taskId and action[2]==segId): continue
            new_avail_schedules.append(action)
        self.avail_schedules = new_avail_schedules

        return 0

    def query_state(self) -> 'tuple':
        self.current_time = self.client.get_current_time_stamp()
        self.proc_states: 'tuple' = self.client.query_processor_states()
        for i in range(5):
            if self.self_suspension:
                self.task_state[i] = self.client.query_ss_task_state(i)
            else:
                self.task_state[i] = self.client.query_task_state(i)
        
        return (self.current_time, self.proc_states, tuple(self.task_state), tuple(self.to_schedule))
    
    def query_state_lazy(self) -> 'tuple':
        return (self.current_time, self.proc_states, tuple(self.task_state), tuple(self.to_schedule))

    def is_terminated(self) -> bool:
        if self.client.does_task_miss_deadline(): return True
        if self.client.is_simulation_completed(): return True
        return False

    def update_time(self) -> 'tuple[int, bool]':
        """advance the simulator by 1 time

        Returns:
            reward (int):  -5000 if miss ddl, 1000 if complete
            terminate (bool): true if (either miss ddl / complete)
        """
        reward = self.client.update_processor_and_task()

        terminate = False
        if self.client.does_task_miss_deadline():
            reward -= 5000
            terminate = True
        elif self.client.is_simulation_completed():
            reward += 500
            terminate = True

        return reward, terminate

    def debug_print(self):
        self.client.print()