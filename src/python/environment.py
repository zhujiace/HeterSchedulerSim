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
        action_space: return tuple of available scheduling  \n
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

        self.task_state = np.zeros(10, dtype=tuple)
        self.avail_actions = []
        self.searched_at = -1

    
    def reset(self):
        del self.client
        self.client = SimulatorClient("../../build/main")

        self.client.create_processor(0, 2)
        self.client.create_processor(7, 2)

        tasks = self.task_generator.generate(self.utilization)
        for task in tasks:
            self.client.create_heter_ss_task(task[0], 2, (0,7), task[1])
        
        self.client.start_simulation()
        self.schedule_plan = np.zeros((5, 5,10), dtype=bool)
    
    def action_space(self) -> list:
        """search and return list of available schedulings

        Returns:
            list[tuple]: (i,j,k), schedule the k-th segment of j-th task on i-th processor
        """
        self.query_state()
        if self.searched_at == self.client.get_current_time_stamp():
            return self.avail_actions

        self.avail_actions = []
        for i, proc_state in enumerate(self.proc_states):
            # ignore non-preemptive procs
            if proc_state[1] > 1: continue
            for j, task_state in enumerate(self.task_state):
                for k,segment in enumerate(task_state[1]):
                    # not the same affinity
                    if segment[0]!=proc_state[0]: continue
                    # already on the processor, useless
                    if segment[1]==i: continue
                    # segment not ready yet
                    if segment[2]==0: continue
                    # segment is already completed
                    if segment[4]==0: continue
                    self.avail_actions.append((i, j, k))

        self.searched_at = self.client.get_current_time_stamp()

        return self.avail_actions

    def step(self, *args) -> 'tuple[tuple, int, bool, dict]':
        if (len(args) == 0):
            return self.update_time()
        return self.schedule(args[0], args[1], args[2])

    def schedule(self, procId:int, taskId: int, segId: int) -> 'tuple[tuple, int, bool, dict]':
        """Perform the schedule command. The avail action is updated.

        Returns:
            state (tuple): (time, <pState>, <tStates>)\n
            reward (int):  1 if schedule a task, -10 if wrong behavior
            terminate (bool): False (must be false)
            info (dict): used for future debugging
        """
        if (not (procId, taskId, segId) in self.avail_actions):
            return self.query_state_lazy(), -10, False, {"valid": False}
        res = self.client.schedule_segment_on_processor(procId, taskId, segId)
        if res.find("Error")!=-1 :
            return self.query_state_lazy(), -10, False, {"valid": False}

        new_avail_actions = []
        for action in self.avail_actions:
            if (action[0]==procId): continue
            if (action[1]==taskId and action[2]==segId): continue
            new_avail_actions.append(action)
        self.avail_actions = new_avail_actions

        return self.query_state(), 1, False, {}

    def query_state(self) -> 'tuple':
        self.current_time = self.client.get_current_time_stamp()
        self.proc_states: 'tuple' = self.client.query_processor_states()
        for i in range(5):
            self.task_state[i] = self.client.query_task_state(i)
        
        return (self.current_time, self.proc_states, tuple(self.task_state))
    
    def query_state_lazy(self) -> 'tuple':
        return (self.current_time, self.proc_states, tuple(self.task_state))

    def is_terminated(self) -> bool:
        if self.client.does_task_miss_deadline(): return True
        if self.client.is_simulation_completed(): return True
        return False

    def update_time(self) -> 'tuple[tuple, int, bool, dict]':
        """advance the simulator by 1 time

        Returns:
            state (tuple): (time, <pState>, <tStates>)\n
            reward (int):  -5000 if miss ddl, 1000 if complete
            terminate (bool): true if (either miss ddl / complete)
            info (dict): should be empty by default
        """
        self.update_time()

        reward = 0
        terminate = False
        if self.client.does_task_miss_deadline():
            reward = -5000
            terminate = True
        elif self.client.is_simulation_completed():
            reward = 500
            terminate = True

        return self.query_state(), reward, terminate, {}
