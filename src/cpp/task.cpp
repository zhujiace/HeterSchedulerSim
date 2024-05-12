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
    return true;
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
        if (task.queryTaskState()==SegmentState_t::SEG_READY)
            return task;
    return internalTasks[0];
}


double HeterSSTask::queryTaskUtilization() {
    double utilizationSum = 0.0;
    for (Task & task : internalTasks)
        utilizationSum += double(taskPeriod) / task.querySegmentExecutionTime();
    return utilizationSum;
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
    heterSSTaskState = HeterSSTaskState_t::TASKS_READY;
    return true;
}

bool HeterSSTask::isAllTasksCompleted() {
    for (Task & task : internalTasks)
        if (!task.isTaskCompleted()) return false;
    heterSSTaskState = HeterSSTaskState_t::TASKS_FINISHED;
    return true;
}

bool HeterSSTask::checkWhetherMissDDL(TimeStamp_t currentTime) {
    bool res = (currentTime > taskAbsoluteDeadline);
    if (res) heterSSTaskState = TASKS_MISSDDL;
    return res;
}


HeterSSTaskState_t HeterSSTask::queryHeterSSTaskState() {
    return heterSSTaskState;
}

SegmentLength_t Task::querySegmentExecutionTime() {
    SegmentLength_t res = 0;
    for (Segment & seg : segments)
        res += seg.querySegmentLength();
    return res;
}

bool Task::isTaskCompleted() {
    for (Segment & seg : segments)
        if (!seg.isSegmentCompleted()) return false;
    return true;
}

bool Task::executeSegment(SegmentIndex_t segmentIndex, TimeStamp_t timeStamp) {
    return segments[segmentIndex].executeSegment(timeStamp);
}

bool Task::executeFirstReadySegment(TimeStamp_t timeStamp) {
    for (Segment & seg : segments)
        if (seg.isSegmentReady())
            return seg.executeSegment(timeStamp);
    return false;
}

bool Task::createNewSegment(SegmentLength_t segmentLength) {
    unsigned int segmentCount = segments.size();
    segments.push_back(Segment(segmentLength, segmentCount, taskPreemption));
    return true;
}


bool Task::_resetAllSegments() {
    for (Segment & seg : segments)
       if (!seg.resetSegment()) return false;
    return true;
}

bool Task::isInsideProcessorMasks(processor::ProcessorIndex_t processorGlobalIndex) {
    for (ProcessorIndex_t index : processorMasks)
        if (processorGlobalIndex == index)
            return true;
    return false;
}

SegmentState_t Task::queryTaskState() {
    SegmentState_t res = SegmentState_t::SEG_NOTREADY;
    for (Segment & seg: segments)
        if (seg.isSegmentReady()) return SegmentState_t::SEG_READY;
    if (isTaskCompleted()) return SegmentState_t::SEG_FINISHED;
    return res;
}

void Task::setSegmentDependency(SegmentIndex_t segment1, SegmentIndex_t segment2) {
    segments[segment2].addToDependency(segments[segment1]);
}

bool Task::setTaskScheduled() {
    belongedHeterSSTask->setHeterSSTaskState(HeterSSTaskState_t::TASKS_EXECUTING);
    return true;
}

bool Task::setTaskPreempted() {
    belongedHeterSSTask->setHeterSSTaskState(HeterSSTaskState_t::TASKS_READY);
    return true;
}

void Segment::addToDependency(Segment & segment) {
    dependentSegments.push_back(&segment);
    segment.dependentedBySegments.push_back(this);
}

bool Segment::isSegmentReady() {
    for (unsigned int i = 0; i < dependentSegments.size(); i++)
        if (!dependentSegments[i]->isSegmentCompleted()) return false;
    markSegmentReady();
    return true;
}

bool Segment::resetSegment() {
    executedAt.clear();
    if (segmentRemainLength !=0) return false;
    segmentRemainLength = segmentLength;
    return true;
}

bool Segment::executeSegment(TimeStamp_t timeStamp) {
    if (segmentRemainLength<=0) return false;
    if (!executedAt.empty() && executedAt.back()+1!=timeStamp) return false;
    segmentRemainLength--;
    executedAt.push_back(timeStamp);
    return true;
}
