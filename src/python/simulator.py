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
            stderr=subprocess.STDOUT,
            text=True
        )

    def send_command(self, command):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        return self.process.stdout.readline().strip()

    def get_current_time_stamp(self):
        return self.send_command("queryCurrentTimeStamp    ")
    
    def quit(self):
        return self.send_command("quit   ")

    def close(self):
        self.process.stdin.close()
        self.process.stdout.close()
        self.process.stderr.close()
        self.process.wait()

if __name__ == "__main__":
    simulator = SimulatorClient("/home/hamster/HeterSchedulerSim/build/main")
    print(simulator.get_current_time_stamp())
    print(simulator.get_current_time_stamp())
    print(simulator.get_current_time_stamp())
    print(simulator.get_current_time_stamp())
    print(simulator.get_current_time_stamp())
    print(simulator.quit())
