import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time

def run_command(cmd):
    """run single program"""
    process = subprocess.Popen(cmd, shell=True)
    process.communicate()
    return process.returncode

def manage_processes(commands, max_parallel):
    """Keep max_parallel program running"""
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_cmd = {executor.submit(run_command, cmd): cmd for cmd in commands[:max_parallel]}
        commands = commands[max_parallel:]

        while future_to_cmd:
            # ease cpu usage
            time.sleep(10)
            
            for future in as_completed(future_to_cmd):
                cmd = future_to_cmd.pop(future)
                try:
                    future.result()
                except Exception as e:
                    print(f"{cmd} 运行时出现错误: {e}")
                
                if commands:
                    next_cmd = commands.pop(0)
                    future_to_cmd[executor.submit(run_command, next_cmd)] = next_cmd

def main():
    with open('commands.txt', 'r') as file:
        commands = [line.strip() for line in file.readlines() if line.strip()]

    num_cpus = os.cpu_count()
    
    max_parallel = min(80, num_cpus)
    
    manage_processes(commands, max_parallel)

if __name__ == "__main__":
    main()
