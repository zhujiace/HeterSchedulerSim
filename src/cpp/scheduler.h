/*
Copy Right. The EHPCL Authors.
*/

#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "processor.h"


/**
 * @brief The simulator (single-instance mode) exposed to users.
 * The simulators should support the following functions:
 * 1. initialization on processors and tasksets,
 * 2. schedule command on tasks,
 * 3. query task and processor states
 * 
*/
class Simulator {

    std::vector<Processor> processors = {};

    std::vector<HeterSSTask> heterSSTaskset = {};

    //TODO: support more task types by adding other attributes.
public:

    /**
     * @brief Create one new processor and insert in the vector end,
     * set the temperate global index as vector index.
     * @return True if successfully created.
    */
    bool createNewProcessor(processor::ProcessorType_t processorType);
    bool createNewProcessors(processor::ProcessorType_t processorType, unsigned int processorCount);
    bool sortProcessorsByType();

    /**
     * @brief Create one new empty task and insert in the vector end,
     * without initilize the dependencies.
     * @return The handle of the empty task.
    */
    HeterSSTask & createNewHeterSSTask();
    HeterSSTask & createNewHeterSSTaskWithVector(std::vector<unsigned int> segments);

    ProcessorState_t queryProcessorState(ProcessorIndex_t processorGlobalIndex);
    HeterSSTaskState_t queryHeterSSTaskState(HeterTaskIndex_t heterTaskIndex);

    Processor & getProcessor(ProcessorIndex_t processorGlobalIndex) 
        {return processors[processorGlobalIndex];};
    HeterSSTask & getHeterSSTask(HeterTaskIndex_t heterTaskIndex)
        {return heterSSTaskset[heterTaskIndex];};
};

class Scheduler {

    Simulator simulator;

    TaskRTSchedulePolicy_t schedulePolicy = HETER_SCHED_FIFO;

public:
    bool schedule(Task task);

    Simulator & getSimulator() {return simulator;};
};

#endif // scheduler.h
