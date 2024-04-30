/*
Copy Right. The EHPCL Authors.
*/

#ifndef PROCESSOR_H
#define PROCESSOR_H

#include <string>

#include "affinity.h"

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

class Processor {

    ProcessorType_t processorType; 
    ProcessorState_t processorState;

public:
    ProcessorState_t stateQuery();
    ProcessorType_t typeQuery();

};

#endif // processor.h
