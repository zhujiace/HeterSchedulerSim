/*
Copy Right. The EHPCL Authors.
*/

#ifndef PROCESSOR_H
#define PROCESSOR_H

#include <string>

#include "task.h"

namespace processor {
    
typedef ProcessorAffinity_t ProcessorType_t;

const std::string ProcessorTypeNames[10] = {
    "CPU",
    "CPUBigCore",
    "CPULittleCore",
    "DataCopy",
    "DataCopyHTD",
    "DataCopyDTH",
    "PE",
    "GPU",
    "FPGA",
    "UNKNOWN",
};

enum ProcessorState_t {
    IDLE,
    BUSYPREEMPTIVE,
    BUSYNONPREEMPTIVE,
    DEAD,
    UNKNOWN
};

typedef TaskPreemption_t ProcessorPreemption_t;
typedef unsigned int ProcessorIndex_t;

};

using namespace processor;

class Processor {

protected:
    ProcessorType_t processorType = CPU;
    ProcessorPreemption_t processorPreemption = ProcessorPreemption_t::PREEMPTIVE;
    ProcessorState_t processorState = IDLE;
    ProcessorIndex_t processorGlobalIndex = 0;
    ProcessorIndex_t processorInternalIndex = 0;

    Task * currentTask = nullptr;

    TaskRTPriority_t currentTaskPriority = 99;

public:
    ProcessorType_t queryProcessorType() {return processorType;};
    ProcessorState_t queryProcessorState() {return processorState;};
    ProcessorIndex_t queryProcessorGlobalIndex() {return processorGlobalIndex;};

    TaskRTPriority_t queryProcessorCurrentTaskPriority() {return currentTaskPriority;}

    void setProcessorInternalIndex(ProcessorIndex_t processorInternalIndex)
        {this->processorInternalIndex = processorInternalIndex;}
    void setProcessorGlobalIndex(ProcessorIndex_t processorGlobalIndex)
        {this->processorGlobalIndex = processorGlobalIndex;}

    bool operator<(const Processor & other) const {
        return processorType < other.processorType;
    }

    Task & getCurrentTask() {return *currentTask;};

    /**
     * @brief Schedule (as a decision) the given task on this processor, taking account
     * the processor affinity and preemption property. Though the schedule command is only
     * called from the processor side, the task state is changed internally after the
     * schedule is called.
    */
    bool scheduleTask(Task & taskToSchedule, task::TimeStamp_t timeStamp);

    // Default constructor, create an empty processor.
    Processor() {};

    Processor(ProcessorType_t processorType, ProcessorPreemption_t ProcessorPreemption,
              ProcessorIndex_t processorGlobalIndex):
              processorType(processorType), processorPreemption(ProcessorPreemption),
              processorGlobalIndex(processorGlobalIndex) {};

    bool setProcessorState(ProcessorState_t processorNewState)
        {processorState = processorNewState;};

    // Simulate the behavior: either execute the task or keep idle
    // Update the processor state if necessary
    bool workProcessor(task::TimeStamp_t timeStamp);
};

#endif // processor.h
