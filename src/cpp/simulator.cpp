/*
Copy Right. The EHPCL Authors.
*/

#include <algorithm>

#include "simulator.h"

ProcessorPreemption_t Simulator::queryProcessorPreemptionBasedonType(ProcessorType_t processorType) {
    if (processorType==CPU) return ProcessorPreemption_t::PREEMPTIVE;
    if (processorType==CPUBigCore) return ProcessorPreemption_t::PREEMPTIVE;
    if (processorType==CPULittleCore) return ProcessorPreemption_t::PREEMPTIVE;
    return ProcessorPreemption_t::NONPREEMPTIVE;
}

bool Simulator::createNewProcessor(processor::ProcessorType_t processorType) {
    unsigned int currentProcessorNum = processors.size();
    ProcessorPreemption_t processorPreemption = queryProcessorPreemptionBasedonType(processorType);
    processors.push_back(Processor(processorType, processorPreemption, currentProcessorNum));
    return true;
}

bool Simulator::createNewProcessors(processor::ProcessorType_t processorType, 
                                    unsigned int processorCount) {
    for (unsigned int i = 0 ; i < processorCount; i++)
        if (!createNewProcessor(processorType)) return false;
    return true;
}

bool Simulator::sortProcessorsByType() {
    std::sort(processors.begin(), processors.end(), std::less<Processor>());
    for (unsigned int i = 0; i < processors.size(); i++)
        processors[i].setProcessorGlobalIndex(i);
    ProcessorAffinity_t lastType = ProcessorAffinity_t::UNKNOWN;
    unsigned int lastCount = 0;
    for (auto & processor : processors) {
        if (processor.queryProcessorType() != lastType) {
            lastCount = 0;
            lastType = processor.queryProcessorType();
        }
        processor.setProcessorInternalIndex(lastCount++);
    }
    return true;
}

ProcessorState_t Simulator::queryProcessorState(ProcessorIndex_t processorGlobalIndex) {
    return processors[processorGlobalIndex].queryProcessorState();
}

HeterSSTaskState_t Simulator::queryHeterSSTaskState(HeterTaskIndex_t heterTaskIndex) {
    return heterSSTaskset[heterTaskIndex].queryHeterSSTaskState();
}

HeterSSTask & Simulator::createNewHeterSSTask() {
    heterSSTaskset.push_back(HeterSSTask());
    return heterSSTaskset[heterSSTaskset.size()-1];
}

HeterSSTask & Simulator::createNewHeterSSTaskWithVector(std::vector<ProcessorAffinity_t> processorType,
                                                        std::vector<unsigned int> segments) {
    HeterSSTask & result = createNewHeterSSTask();
    result.initializeTaskByVector(processorType, segments);
    return result;
}

void Simulator::checkTaskRelease() {
    if (taskReleaseCheckedThisRound) return;
    for (HeterSSTask & htask: heterSSTaskset) {
        if (htask.queryTaskPeriod()%currentTimeStamp==0) {
            htask.releaseTask(currentTimeStamp);
        }
    }
    taskReleaseCheckedThisRound = true;
}

void Simulator::updateProcessorAndTask() {
    if (!taskReleaseCheckedThisRound) checkTaskRelease();

    for (Processor & processor: processors) {
        processor.workProcessor(currentTimeStamp);
    }

    for (HeterSSTask & htask: heterSSTaskset) {
        if (htask.checkWhetherMissDDL(currentTimeStamp))
            taskMissDeadline = true;
    }

    currentTimeStamp++;
    taskReleaseCheckedThisRound = false;
}

Task & Simulator::queryReadyTask(HeterTaskIndex_t heterTaskIndex) {
    HeterSSTask & htask = heterSSTaskset[heterTaskIndex];
    return htask.getReadyTask();
}

