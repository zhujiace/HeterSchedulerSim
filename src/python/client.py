# 
# Copy Right. The EHPCL Authors.
#

import subprocess

class SimulatorClient:

    def __init__(self, executable_path):
        self.process = subprocess.Popen(
            [executable_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True
        )

    def send_command(self, command):
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
        return bool(self.send_command("isSimulationCompleted"))
    
    def does_task_miss_deadline(self) -> bool:
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
    def create_processor(self, procType: str, procNum: int = 1) -> str:
        pass

    @command_decorator("createHeterSSTask {} {} {} {}")
    def create_heter_ss_task_helper(self, period:int, procCount: int,
                                    procs: str, segs: str) -> str:
        pass

    def create_heter_ss_task(self, period:int, procCount: int,
                             proc: 'tuple[str]', segs: 'tuple[int]') -> str:
        return self.create_heter_ss_task_helper(period, procCount,
                                                " ".join(proc)+" ",
                                                " ".join(map(str, segs))+" ")

    @command_decorator("queryProcessorStates")
    def query_processor_state_helper(self) -> str:
        pass

    def query_processor_state(self) -> 'tuple[int]':
        res = self.query_processor_state_helper()
        return tuple(map(int, res.split()))

    @command_decorator("queryTaskSegmentStates {}")
    def query_task_segment_states_helper(self, taskId: int) -> str:
        pass

    def query_task_segment_states(self, taskId: int) -> 'list[tuple[int]]':
        res = self.query_task_segment_states_helper(taskId)
        resList = list(map(int, res.split()))
        result = []
        for i in range(len(resList)//3):
            result.append((resList[3*i], resList[3*i+1], resList[3*i+2]))
        return result

    @command_decorator("scheduleSegmentOnProcessor {} {} {}")
    def schedule_segment_on_processor(self, procId: int, taskId:int, segId: int) -> str:
        pass

    def print(self):
        return self.send_command("printSimulatorState")

if __name__ == "__main__":
    simulator = SimulatorClient("/home/hamster/HeterSchedulerSim/build/main")
    print(simulator.create_processor("CPU", 2))
    print(simulator.create_heter_ss_task(5,2,("CPU", "GPU"), (1,1,1,1,1)))
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
