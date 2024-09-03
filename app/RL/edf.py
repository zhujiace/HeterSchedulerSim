
from environment import SimulationEnv

from numpy import random

random.seed(179)

avail_seeds = []

for seeds in range(6, 7):

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
            furtherestDeadline = -1
            for j in range(2):
                if int(cpuStates[j])==0:
                    decisions[cpuTaskList[i]] = j+1
                    cpuStates[j] = 10
                    break
                elif int(cpuStates[j])==1:
                    if deadline[int(state[j*4+3])] > deadline[cpuTaskList[i]]:
                        if deadline[int(state[j*4+3])] > furtherestDeadline:
                            furtherestDeadline = deadline[int(state[j*4+3])]
                            decisions[cpuTaskList[i]] = j+1

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
        avail_seeds.append(seeds)
        print(f"current seed: {seeds}")
        # break

from sys import stderr
print(avail_seeds, file=stderr)


# [6, 8, 14, 28, 34, 36, 41, 50, 68, 86, 88, 96, 108, 109, 111, 122, 128, 129, 146, 150, 151, 153, 154, 161, 168, 174, 182, 196, 205, 206, 214, 217, 225, 228, 231, 234, 245, 246, 274, 288, 289, 294, 306, 314, 321, 328, 336, 345, 346, 348, 350, 362, 368, 374, 385, 388, 391, 396, 407, 408, 416, 428, 429, 430, 435, 436, 441, 448, 450, 451, 465, 468, 471, 474, 482, 489, 491, 494, 496, 501, 506, 509, 511, 514, 516, 525, 526, 528, 529, 534, 541, 548, 549, 554, 565, 574, 576, 588, 591, 596, 610, 628, 634, 636, 642, 648, 651, 654, 656, 665, 688, 694, 725, 731, 735, 741, 746, 748, 756, 768, 771, 774, 781, 785, 791, 805, 808, 826, 842, 846, 848, 851, 854, 862, 868, 871, 906, 908, 910, 914, 916, 928, 946, 951, 954, 968, 981, 985, 991]