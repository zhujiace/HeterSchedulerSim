# 
# Copy Right. The EHPCL Authors.
#

import subprocess

class SimulatorClient:
    """ Python client for interecting with C++ backend

    Examples
    --------
    >>> sim = SimulatorClient("path_to_C++_executable")
    >>> sim.startSimulation()
    """

    def __init__(self, executable_path):
        self.process = subprocess.Popen(
            [executable_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True
        )
        self.procMap = {0: "CPU", 7: "GPU"}

    def send_command(self, command: str):
        """send command in str to the C++ process

        Returns:
            str: striped stdout from the C++ backend
        """
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        return self.process.stdout.readline().strip()


    def command_decorator(command_template: str):
        from functools import wraps
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs) -> str:
                command = command_template.format(*args, **kwargs)
                return self.send_command(command)
            return wrapper
        return decorator

    def get_current_time_stamp(self) -> int:
        return int(self.send_command("queryCurrentTimeStamp"))
    
    @command_decorator("quit")
    def quit(self) -> str:
        pass
    
    @command_decorator("setSimulationTimeBound {}")
    def set_simulation_timebound(self, bound: int) -> str:
        pass

    def is_simulation_completed(self) -> bool:
        """return true is the simulation have reached the limit"""
        return bool(self.send_command("isSimulationCompleted"))
    
    def does_task_miss_deadline(self) -> bool:
        """return true if there's any task miss deadline,
        the agent should stop simulation
        """
        return bool(self.send_command("doesTaskMissDeadline"))
    
    @command_decorator("updateProcessorAndTask")
    def update_processor_and_task(self) -> str:
        pass

    @command_decorator("sortProcessors")
    def sort_processors(self) -> str:
        pass

    @command_decorator("startSimulation")
    def start_simulation(self) -> str:
        pass

    @command_decorator("createProcessor {} {}")
    def _create_processor_helper(self, procType: str, procNum: int = 1) -> str:
        pass

    def create_processor(self, procType: int, procNum: int = 1) -> str:
        """create processors in the simulator

        Args:
            procType (int): 0 -> CPU, 7 -> GPU
            procNum (int, optional): Defaults to 1.

        Returns:
            str: Created Successfully
        """
        return self._create_processor_helper(self.procMap[procType], procNum)

    @command_decorator("createHeterSSTask {} {} {} {}")
    def _create_heter_ss_task_helper(self, period:int, procCount: int,
                                    procs: str, segs: str) -> str:
        pass

    def create_heter_ss_task(self, period:int, procCount: int,
                             proc: 'tuple[int]', segs: 'tuple[int]') -> str:
        """create a heterogenous self-suspension based task in the simulator

        Args:
            period (int): period of the task
            procCount (int): number of different processor involved
            proc (tuple[int]): e.g. (CPU, GPU) -> (0,7)
            segs (tuple[int]): segments

        Examples:
        >>> (5, 2, (0,7), (1,1,1,1,1))
        """
        return self._create_heter_ss_task_helper(period, procCount,
                                                " ".join([self.procMap[x] for x in proc]) + " ",
                                                " ".join(map(str, segs))+" ")

    @command_decorator("queryProcessorStates")
    def _query_processor_state_helper(self) -> str:
        pass

    def query_processor_state(self) -> 'tuple[int]':
        res = self._query_processor_state_helper()
        return tuple(map(int, res.split()))

    @command_decorator("queryTaskSegmentStates {}")
    def _query_task_segment_states_helper(self, taskId: int) -> str:
        pass

    def query_task_segment_states(self, taskId: int) -> 'list[tuple[int]]':
        res = self._query_task_segment_states_helper(taskId)
        resList = list(map(int, res.split()))
        result = []
        for i in range(len(resList)//4):
            result.append((resList[4*i], resList[4*i+1], resList[4*i+2], resList[4*i+3]))
        return result
    
    def query_task_states(self) -> 'list[int]':
        result = list(map(int, self.send_command("queryTaskStates").split()))
        return result

    @command_decorator("scheduleSegmentOnProcessor {} {} {}")
    def schedule_segment_on_processor(self, procId: int, taskId:int, segId: int) -> str:
        pass

    def print(self):
        return self.send_command("printSimulatorState")

if __name__ == "__main__":
    simulator = SimulatorClient("/home/hamster/HeterSchedulerSim/build/main")
    CPU = 0; GPU = 7
    print(simulator.create_processor(CPU, 2))
    print(simulator.create_processor(GPU, 1))
    print(simulator.create_heter_ss_task(5,2,(CPU, GPU), (1,1,1,1,1)))
    print(simulator.start_simulation())
    print(simulator.print())
    print(simulator.schedule_segment_on_processor(0,0,0))
    print(simulator.print())
    print(simulator.update_processor_and_task())
    print(simulator.update_processor_and_task())
    print(simulator.update_processor_and_task())
    print(simulator.schedule_segment_on_processor(1, 0, 1))
    print(simulator.update_processor_and_task())
    print(simulator.print())
    print(simulator.update_processor_and_task())
    print(simulator.print())
    print(simulator.update_processor_and_task())
    print(simulator.does_task_miss_deadline())
    simulator.quit()
