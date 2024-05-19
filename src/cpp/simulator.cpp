/*
Copy Right. The EHPCL Authors.
*/

#include <algorithm>
#include <iostream>

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

TaskState_t Simulator::queryTaskState(TaskIndex_t taskIndex) {
    return taskset[taskIndex].queryTaskState();
}

Task & Simulator::createNewTask() {
    taskset.push_back(Task());
    taskset.back().setTaskIndex(taskset.size()-1);
    return taskset.back();
}

Task & Simulator::createNewHeterSSTaskWithVector(std::vector<ProcessorAffinity_t> processorType,
                                                        std::vector<unsigned int> segments) {
    Task & result = createNewTask();
    result.initializeTaskByVector(processorType, segments);
    return result;
}

bool Simulator::checkTaskRelease() {
    if (taskReleaseCheckedThisRound) return true;
    for (Task & task: taskset) {
        if (currentTimeStamp%task.queryTaskPeriod()==0) {
            if (!task.releaseTask(currentTimeStamp))
                return false;
        }
    }
    return (taskReleaseCheckedThisRound = true);
}

void Simulator::updateProcessorAndTask() {
    if (!taskReleaseCheckedThisRound) checkTaskRelease();

    for (Processor & processor: processors) {
        if (!processor.workProcessor(currentTimeStamp)) {
            std::cout << "Processor working error!";
        }
    }

    for (Task & task: taskset) {
        task.checkTaskStates();
        if (task.checkWhetherMissDDL(currentTimeStamp))
            taskMissDeadline = true;
    }

    currentTimeStamp++;
    taskReleaseCheckedThisRound = false;
}


void Simulator::initializeStorages() {
    taskset.reserve(10);
    processors.reserve(10);
}

static std::ostream & operator<<(std::ostream & os, const Processor & processor) {
    os << std::string("State: ");
    switch (processor.queryProcessorState()) {
        case IDLE:
            os << std::string("idle");break;
        case BUSY_PREEMPTIVE:
            os << std::string("busy-preemptive");break;
        case BUSY_NONPREEMPTIVE:
            os << std::string("busy-nonpreemptive");break;
        case DEAD:
            os << std::string("dead");break;
        default:
            os << std::string("unknown");break;
    }
    os << std::string(", Task ");
    if (processor.getCurrentTask())
    os << std::to_string(processor.getCurrentTask()->queryTaskIndex());
    os << std::string(", Segment: ");
    if (processor.getCurrentSegment())
    os << std::to_string(processor.getCurrentSegment()->querySegmentLength() - processor.getCurrentSegment()->querySegmentRemainLength())
       << std::string("/") << std::to_string(processor.getCurrentSegment()->querySegmentLength());
    return os;
}


void Simulator::printSimulatorStates() {
    std::cerr << "Current Timestamp: " << currentTimeStamp << std::endl;
    unsigned int count = 0;
    for (Task & task : taskset) {
        std::cerr <<  task << std::endl;
    }
    count = 0;
    for (Processor & processor: processors) {
        std::cerr << "Processor " << count++ << " " << processor << std::endl;
    }
    std::cerr << std::endl;
}

