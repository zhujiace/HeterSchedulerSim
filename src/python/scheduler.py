
from client import SimulatorClient

if __name__ == "__main__":
    # default scheduler example
    simulator = SimulatorClient("/home/hamster/HeterSchedulerSim/build/main")

    # Step 1: initilization
    simulator.create_processor("CPU", 2)
    simulator.create_processor("GPU", 2)

    # create 5 tasks
    for i in range(5):
        filename = f"/home/hamster/HeterSchedulerSim/test/.taskset_prop/task{i}.txt"
        with open(filename) as taskfile:
            period: int = int(taskfile.readline().strip())
            line = taskfile.readline().strip()
            segs = tuple(map(int, line.split()))
            simulator.create_heter_ss_task(period, 2, ("CPU", "GPU"), segs)
    simulator.set_simulation_timebound(200)
    simulator.start_simulation()
    simulator.print()

    # step 2: scheduling loop
    while (not simulator.is_simulation_completed()):
        #schedule decision here

        """
        // First round: check IDLE Processors
        for (ProcessorIndex_t j = 0; j < simulator.queryProcessorCount(); j++) {
        Processor & proc = simulator.getProcessor(j);
        if (proc.queryProcessorState()!=IDLE) continue;
        for (TaskIndex_t i = 0; i < simulator.queryTaskCount(); i++) {
            Segment * readySegmet = simulator.getTask(i).
                                    getFirstReadySegment(proc.queryProcessorType());
            if (!readySegmet) continue;
            proc.scheduleTaskSpecifiedSegment(simulator.getTask(i), readySegmet, simulator.queryCurrentTimeStamp());
            break;
        }
        }
        """
        procState = simulator.query_processor_state()
        for i in range(4):
            if procState[i] != 0: continue
            for j in range(5):
                segState = simulator.query_task_segment_states(j)
                for k, seg in enumerate(segState):
                    isReady = seg[0]; currentProc = seg[2]
                    if (isReady==1 and currentProc==999999):
                        if (k%2==1 and i <= 1): continue
                        if (k%2==0 and i >= 2): continue
                        simulator.schedule_segment_on_processor(i, j, k)
                procState = simulator.query_processor_state()
                if procState[i] != 0: break

        simulator.update_processor_and_task()
        simulator.print()

        a = input("Wait for your response")

    simulator.quit()

