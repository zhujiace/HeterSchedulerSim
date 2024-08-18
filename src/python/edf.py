from environment import SimulationEnv

from numpy import random

random.seed(179)

for seeds in range(13, 10000):

    env = SimulationEnv(seeds, utilization=2.3)

    state = env.reset()
    # 15, 20, 30, 45, 180
    period = [state[17 + 14 * i] for i in range(5)] 
    done = False

    idle_schedule_count = 0

    while not done:

        # get the time stamp
        current_time = env.current_time
        # calculate the deadlines
        deadline = [period[i] - current_time%period[i] for i in range(5)]
        if current_time == 30:
            print(f"deadline: {deadline}")
        sorted_indices = sorted(range(len(deadline)), key=lambda x: deadline[x])

        to_schedule = [True for _ in range(5)]
        for i in range(5):
            if int(state[19+14*i])!=-1:
                to_schedule[i] = False

        cpuTaskList = []
        gpuTaskList = []
        for tskIndex in sorted_indices:
            if to_schedule[tskIndex]==False:
                continue
            if int(state[18+14*tskIndex])%2==0:
                if len(cpuTaskList) < 2:
                    if int(state[20+14*tskIndex])>0:
                        cpuTaskList.append(tskIndex)
            else:
                if len(gpuTaskList) < 2:
                    if int(state[20+14*tskIndex])>0:
                        gpuTaskList.append(tskIndex)

        cpuStates = [state[2], state[6]]
        gpuStates = [state[10], state[14]]
        
        decisions = [0,0,0,0,0]
        for i in range(len(cpuTaskList)):
            for j in range(2):
                if int(cpuStates[j])==0:
                    decisions[cpuTaskList[i]] = j+1
                    cpuStates[j] = 10
                    break
                elif int(cpuStates[j])==1:
                    if deadline[int(state[j*4+3])] > deadline[cpuTaskList[i]]:
                        decisions[cpuTaskList[i]] = j+1
                        break

        for i in range(len(gpuTaskList)):
            for j in range(2):
                if int(gpuStates[j])==0:
                    decisions[gpuTaskList[i]] = j+3
                    gpuStates[j] = 2
                    break

        print(env.client.send_command("printSimulatorState "))
        print(f"cpuList: {cpuTaskList}")
        print(f"gpuList: {gpuTaskList}")
        print(f"decision: {decisions}")

        while int(state[0])==current_time:
            print(f"make decision! Task {state[-7]}")
            decision = decisions[int(state[-7])]
            state, _ , done, __ = env.step(decision)
            idle_schedule_count += 1 if decision==0 else 0
            # a = input()
            print(env.client.send_command("printSimulatorState "))
            
        # print(f"idle: {idle_schedule_count}")

    # print(idle_schedule_count)

    # 6, 11 ,9 , 23, 19
    # 30, 36, 45, 60, 180
    if env.current_time < 199:
        print(f"current seed: {seeds}")
        break