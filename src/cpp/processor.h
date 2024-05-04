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
    BUSY,
    DEAD,
    NONPREEMPTIVE,
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

public:
    ProcessorType_t queryProcessorType() {return processorType;};
    ProcessorState_t queryProcessorState() {return processorState;};
    ProcessorIndex_t queryProcessorGlobalIndex() {return processorGlobalIndex;};

    void setProcessorInternalIndex(ProcessorIndex_t processorInternalIndex);
    void setProcessorGlobalIndex(ProcessorIndex_t processorGlobalIndex);

    Task & getCurrentTask() {return *currentTask;};

    /**
     * @brief Schedule the given task on this processor, taking account
     * the processor affinity and preemption property.
    */
    bool scheduleTask(Task & taskToSchedule, task::TimeStamp_t timeStamp);

    // Default constructor, create an empty processor.
    Processor() {};

    Processor(ProcessorType_t processorType, ProcessorPreemption_t ProcessorPreemption,
              ProcessorIndex_t processorGlobalIndex):
              processorType(processorType), processorPreemption(ProcessorPreemption),
              processorGlobalIndex(processorGlobalIndex) {};
};

#endif // processor.h
