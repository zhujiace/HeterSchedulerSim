/*
Copy Right. The EHPCL Authors.
*/

#include "task.h"

/**
 * @brief check the task state and update internal storage.
 * @attention different from querying, will detail exam each segments
*/
TaskState_t Task::checkTaskStates() {
    readySegments.clear();
    readySegments.reserve(segments.size());
    TaskState_t res = TASKS_FINISHED;
    for (Segment & seg : segments) {
        if (!seg.isSegmentCompleted()) {
            res = TASKS_UNKNOWN;
            if (seg.isSegmentReady()) {
                res = TASKS_READY;
                readySegments.push_back(&seg);
            }
        }
    }
    if (res == TASKS_UNKNOWN) res = TASKS_FINISHED;
    return res;
}


Segment & Task::createNewSegment(ProcessorAffinity_t processorAffinity, SegmentLength_t segmentLength) {
    if (processorAffinity==CPU || processorAffinity==CPUBigCore || processorAffinity==CPULittleCore)
        this->segments.push_back(Segment(segmentLength, processorAffinity, SegmentPreemption_t::PREEMPTIVE));
    else
        this->segments.push_back(Segment(segmentLength, processorAffinity, SegmentPreemption_t::NONPREEMPTIVE));
    return segments.back();
}

bool Task::initializeTaskByVector(std::vector<ProcessorAffinity_t> & processorType,
                                  std::vector<SegmentLength_t> & segments) {
    this->segments.clear();
    this->segments.reserve(segments.size());

    // Insert segments into the tasks
    // Configure the dependencies
    unsigned int processorTypeCount = processorType.size();
    for (unsigned int i = 0; i < segments.size(); i++) {
        unsigned int processorTypeIndex = i%processorTypeCount;
        ProcessorAffinity_t type = processorType[processorTypeIndex];
        createNewSegment(type, segments[i]);
        if (i!=0) this->segments[i].addToDependency(this->segments[i-1]);
    }
    return true;
}


SegmentLength_t Task::querySegmentExecutionTime() {
    SegmentLength_t totalSegmentLength = 0;
    for (Segment & seg : segments)
        totalSegmentLength += seg.querySegmentLength();
    return totalSegmentLength;
}

double Task::queryTaskUtilization() {
    return querySegmentExecutionTime() / double(taskPeriod);
}

double Task::querySingleTaskUtilization(ProcessorAffinity_t processorAffinity) {
    SegmentLength_t totalSegmentLength = 0;
    for (Segment & seg : segments)
        if (seg.querySegmentProcessorAffinity()==processorAffinity)
            totalSegmentLength += seg.querySegmentLength();
    return totalSegmentLength / double(taskPeriod);
}


bool Task::releaseTask(TimeStamp_t currentTime) {
    if (!resetTask()) return false;
    
    setFirstSegmentReady();
    taskAbsoluteDeadline = taskPeriod + currentTime;
    this->taskState = TaskState_t::TASKS_READY;
    return true;
}


bool Task::checkWhetherMissDDL(TimeStamp_t currentTime) {
    if (taskState == TASKS_FINISHED) return false;
    bool res = (currentTime > taskAbsoluteDeadline);
    if (res) taskState = TASKS_MISSDDL;
    return res;
}


bool Task::isAllSegmentsCompleted() {
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

void Task::setSegmentDependency(SegmentIndex_t segment1, SegmentIndex_t segment2) {
    segments[segment2].addToDependency(segments[segment1]);
}

bool Task::setTaskScheduled() {
    setTaskState(TaskState_t::TASKS_EXECUTING);
    return true;
}

bool Task::setTaskPreempted() {
    setTaskState(TaskState_t::TASKS_READY);
    return true;
}

std::vector<Segment *> & Task::getReadySegments() {
    readySegments.clear();
    for (Segment & seg : segments)
        if (seg.isSegmentReady()) readySegments.push_back(&seg);
    return readySegments;
}

Segment * Task::getFirstReadySegment(ProcessorAffinity_t processorAffinity) {
    getReadySegments();
    for (Segment * & seg : readySegments)
        if (seg->querySegmentProcessorAffinity()==processorAffinity)
            if (!seg->getCurrentProcessor())
                return seg;
    return nullptr;
}

void Segment::addToDependency(Segment & segment) {
    precedingSegments.push_back(&segment);
    segment.succeedingSegments.push_back(this);
}

bool Segment::isSegmentReady() {
    if (isSegmentCompleted()) return false;
    for (Segment * & seg: precedingSegments)
        if (!seg->isSegmentCompleted()) return false;
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
    if (segmentPreemption==SegmentPreemption_t::NONPREEMPTIVE)
    if (!executedAt.empty() && executedAt.back()+1!=timeStamp) return false;
    segmentRemainLength--;
    executedAt.push_back(timeStamp);
    if (segmentRemainLength == 0) {
        segmentCompleted = true;
        segmentReady = false;
        for (Segment * & seg : succeedingSegments) {
            seg->markSegmentReady();
        }
    }
    return true;
}
