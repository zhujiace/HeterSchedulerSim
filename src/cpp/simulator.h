/*
Copy Right. The EHPCL Authors.
*/

#ifndef SIMULATOR_H
#define SIMULATOR_H

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

    TimeStamp_t currentTimeStamp = 0;

    TimeStamp_t maximumSimulationTime = 2147483647L;

    bool taskMissDeadline = false;

    bool taskReleaseCheckedThisRound = false;

    //TODO: support more task types by adding other attributes.
public:

    /**
     * @brief Create one new processor and insert in the vector end,
     * set the temperate global index as vector index.
     * @return True if successfully created.
    */
    bool createNewProcessor(processor::ProcessorType_t processorType);
    bool createNewProcessors(processor::ProcessorType_t processorType, unsigned int processorCount);
    // The sort will change the global index
    bool sortProcessorsByType();

    /**
     * @brief Create one new empty task and insert in the vector end,
     * without initilize the dependencies.
     * @return The handle of the empty task.
    */
    HeterSSTask & createNewHeterSSTask();
    HeterSSTask & createNewHeterSSTaskWithVector(std::vector<ProcessorAffinity_t> processorType, std::vector<unsigned int> segments);
    
    unsigned int queryProcessorCount() {return processors.size();}
    unsigned int queryHeterSSTaskCount() {return heterSSTaskset.size();}

    ProcessorState_t queryProcessorState(ProcessorIndex_t processorGlobalIndex);
    HeterSSTaskState_t queryHeterSSTaskState(HeterTaskIndex_t heterTaskIndex);

    Task & queryReadyTask(HeterTaskIndex_t heterTaskIndex);
    void sortHeterSSTasksByPriority() {/*TODOO*/};

    Processor & getProcessor(ProcessorIndex_t processorGlobalIndex) 
        {return processors[processorGlobalIndex];};
    HeterSSTask & getHeterSSTask(HeterTaskIndex_t heterTaskIndex)
        {return heterSSTaskset[heterTaskIndex];};


    TimeStamp_t queryCurrentTimeStamp() {return currentTimeStamp;};

    /**
     * @brief Check and release the periodic tasks according to current timestamp.
     * @attention This action should be called before making scheduling.
    */
    void checkTaskRelease();

    /**
     * @brief Update all the processors and tasks by calling highLevel APIs.
     * Count the simulator time stamp by one.
     * @attention This action should be called after the scheduling decisions are done.
     * 
     * Task release will be automatically checked.
    */
    void updateProcessorAndTask();

    void setSimulationTimeBound(TimeStamp_t simulationBound) 
        {maximumSimulationTime = simulationBound;};

    bool isSimulationCompleted() {return currentTimeStamp>=maximumSimulationTime;};
    bool doesTaskMissDeadline() {return taskMissDeadline;};

    ProcessorPreemption_t queryProcessorPreemptionBasedonType(ProcessorType_t processorType);
};


#endif // simulator.h
