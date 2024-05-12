/*
Copy Right. The EHPCL Authors.
*/

#include "task.h"

bool HeterSSTask::initializeTaskByVector(std::vector<ProcessorAffinity_t> processorType,
                                        std::vector<SegmentLength_t> segments) {
    internalTasks.clear();
    // Initialize each sub tasks
    for (ProcessorAffinity_t types : processorType) {
        if (types==CPU || types==CPUBigCore || types==CPULittleCore)
            createNewRTTask(types, TaskPreemption_t::PREEMPTIVE);
        else
            createNewRTTask(types, TaskPreemption_t::NONPREEMPTIVE);
    }
    // Insert segments into the tasks
    unsigned int processorTypeCount = processorType.size();
    for (unsigned int i = 0; i < segments.size(); i++) {
        unsigned int processorTypeIndex = i%processorTypeCount;
        createNewSegmentForTask(processorType[processorTypeIndex], segments[i]);
    }
    // Configure the dependencies
    for (ProcessorAffinity_t types : processorType) {
        Task & task = getTask(types);
        unsigned int segCount = task.querySegmentCount();
        for (unsigned int i = 1; i < segCount; i++)
            task.setSegmentDependency(i-1, i);
    }
}

bool HeterSSTask::createNewSegmentForTask(ProcessorAffinity_t processorAffinity,
                                          SegmentLength_t segmentLength) {
    for (Task & task : internalTasks) {
        if (task.queryProcessorAffinity()==processorAffinity) {
            task.createNewSegment(segmentLength);
            return true;
        }
    }
    return false;
}


bool HeterSSTask::createNewRTTask(ProcessorAffinity_t processorAffinity, TaskPreemption_t taskPreemption) {
    return _createNewTask(HARDRT, processorAffinity, taskPreemption);
}

bool HeterSSTask::_createNewTask(TaskRTProperty_t taskRealTimeProperty, 
                        ProcessorAffinity_t processorAffinity, TaskPreemption_t taskPreemption) {
    internalTasks.push_back(Task(taskRealTimeProperty, processorAffinity, taskPreemption));
    internalTasks.back().setBelongedHeterSSTaskset(this);
    return true;
}

Task & HeterSSTask::getTask(ProcessorAffinity_t processorAffinity) {
    for (Task & task : internalTasks)
        if (task.queryProcessorAffinity()==processorAffinity)
            return task;
    return internalTasks[0];
}

Task & HeterSSTask::getReadyTask() {
    for (Task & task : internalTasks)
        if (task.queryTaskState()==SegmentState_t::READY)
            return task;
    return internalTasks[0];
}


double HeterSSTask::queryTaskUtilization() {
    double utilizationSum = 0.0;
    for (Task & task : internalTasks)
        utilizationSum += double(taskPeriod) / task.querySegmentExecutionTime();
}

double HeterSSTask::querySingleTaskUtilization(ProcessorAffinity_t processorAffinity) {
    Task & task = getTask(processorAffinity);
    return double(taskPeriod) / task.querySegmentExecutionTime();
}


bool HeterSSTask::releaseTask(TimeStamp_t currentTime) {
    for (Task & task : internalTasks) 
        task.resetTask();
    
    internalTasks[0].setFirstSegmentReady();
    taskAbsoluteDeadline = taskPeriod + currentTime;
    heterSSTaskState = HeterSSTaskState_t::READY;
}

bool HeterSSTask::isAllTasksCompleted() {
    for (Task & task : internalTasks)
        if (!task.isTaskCompleted()) return false;
    heterSSTaskState = HeterSSTaskState_t::FINISHED;
    return true;
}

bool HeterSSTask::checkWhetherMissDDL(TimeStamp_t currentTime) {
    bool res = (currentTime > taskAbsoluteDeadline);
    if (res) heterSSTaskState = MISSDDL;
    return res;
}


HeterSSTaskState_t HeterSSTask::queryHeterSSTaskState() {
    return heterSSTaskState;
}
