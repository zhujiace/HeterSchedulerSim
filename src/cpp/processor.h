/*
Copy Right. The EHPCL Authors.
*/

#ifndef PROCESSOR_H
#define PROCESSOR_H

#include <string>

#include "task.h"

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

class Processor {

    ProcessorType_t processorType;
    ProcessorPreemption_t processorPreemption;
    ProcessorState_t processorState;
    ProcessorIndex_t processorGlobalIndex;
    ProcessorIndex_t processorInternalIndex;

    Task * currentTask;

public:
    ProcessorState_t queryProcessorState();
    ProcessorType_t queryProcessorType();
    ProcessorIndex_t queryProcessorGlobalIndex();

    Task & getCurrentTask();
};

#endif // processor.h
