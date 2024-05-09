#include "scheduler.h"
#include <fstream>
#include <iostream>
    
bool Scheduler::initializeSimulation() {

    // Test case: 2 CPUs + 2 GPUs

    // Step 1: Initilize processors and tasks

    simulator.createNewProcessors(CPU, 2);
    simulator.createNewProcessors(GPU, 2);
    // It is recommended to sort in case create in a wrong order
    simulator.sortProcessorsByType();

    std::ifstream input_file;
    std:: cout << "Reading taskset infomation...\n";
    for (int i = 0; i < 5; i++) {
        std::string filename = ".taskset_prop/task";
        // TODO: support task num > 10
        filename = filename + (char)(i+48);
        filename = filename + ".txt";
        input_file.open(filename);
        int deadlines;
        input_file >> deadlines;
        std::vector<unsigned int> segments;
        for (int j = 0; j < 2*5-1; j++){
            input_file >> segments[j];
        }
        std::vector<ProcessorAffinity_t> processorTypes = {ProcessorAffinity_t::CPU, ProcessorAffinity_t::GPU};
        HeterSSTask & tmp = simulator.createNewHeterSSTaskWithVector(processorTypes, segments);
        tmp.setTaskRelativeDeadline(deadlines);
        tmp.setTaskPeriod(deadlines);
        tmp.setTaskRTPriority(99-i);
        input_file.close();
    }

    // Step 2: Start simulation
    simulator.setSimulationTimeBound(1000);
    return true;
}

bool Scheduler::startScheduleLoop() {

    while (! simulator.isSimulationCompleted()) {

        simulator.checkTaskRelease();

        // TODO: Make decisions here
        if (!makeScheduleDecisions())
            std::cerr << "Scheduling decisions are inproper!\n";

        simulator.updateProcessorAndTask();
    }

    return (!simulator.doesTaskMissDeadline());

}

bool Scheduler::makeScheduleDecisions() {
    // Cureent Implementation: based on SCHED_FIFO

    simulator.sortHeterSSTasksByPriority();

    for (HeterTaskIndex_t i = 0; i < simulator.queryHeterSSTaskCount(); i++) {
        if (simulator.queryHeterSSTaskState(i)!=HeterSSTaskState_t::READY)
            continue;
        Task & task = simulator.getHeterSSTask(i).getReadyTask();
        Processor * availableProcesser = nullptr;
        for (ProcessorIndex_t j = 0; j < simulator.queryProcessorCount(); j++) {
            if (simulator.getProcessor(j).queryProcessorType()!=task.queryProcessorAffinity())
                continue;
            if (simulator.queryProcessorState(j)==ProcessorState_t::IDLE) {
                availableProcesser = &(simulator.getProcessor(j));
                break;
            }
        }

        // No idle processors and the task is a preemptive task,
        // Search for preemptive processor of this task
        if (!availableProcesser && task.queryTaskPreemption()==TaskPreemption_t::PREEMPTIVE) {
            for (ProcessorIndex_t j = 0; j < simulator.queryProcessorCount(); j++) {
                if (simulator.getProcessor(j).queryProcessorType()!=task.queryProcessorAffinity())
                    continue;
                if (simulator.queryProcessorState(j)==ProcessorState_t::BUSYPREEMPTIVE) {
                    availableProcesser = &(simulator.getProcessor(j));
                    break;
                }
            }
        }

        if (availableProcesser) {
            availableProcesser->scheduleTask(task, simulator.queryCurrentTimeStamp());
        } else {
            // No available processor, do nothing
        }

    }

}