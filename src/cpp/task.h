/*
Copy Right. The EHPCL Authors.
*/

#ifndef TASK_H
#define TASK_H

#include <vector>

#include "affinity.h"
#include "processor.h"

namespace task {

enum TaskPreemption_t {
    PREEMPTIVE,
    NONPREEMPTIVE,
    UNKNOWN
};

enum TaskRTProperty_t {
    HARDRT,
    SOFTRT,
    NONERT,
    UNKNOWN
};

typedef TaskPreemption_t SegmentPreemption_t;
typedef unsigned int SegmentLength_t;
typedef unsigned int SegmentIndex_t;

enum SSTaskState_t {
    EXECUTING,
    SUSPENSION,
    READY,
    UNKNOWN,
};

typedef unsigned long long TimeStamp_t;

};

using namespace task;

/**
 * @brief Describe a segment (SS Task Model) or node (DAG Task Model). 
*/
class Segment {
    SegmentPreemption_t segmentPreemption = PREEMPTIVE;
    SegmentLength_t segmentLength = 0;
    SegmentLength_t segmentRemainLength;
    SegmentIndex_t segmentIndex = 0;

    // The segment can be executed if and only if all the dependent segments are finished.
    unsigned int dependencyCount;
    // The dependent segments may come from other tasks, set to nullptr if none.
    std::vector<Segment *> dependentSegment;

    std::vector<TimeStamp_t> executedAt = {};
public:
    bool isSegmentCompleted() {return segmentRemainLength==0;};
    bool isSegmentReady();
    void addToDependency(Segment & segment);
    bool configureDependency(std::vector<Segment *> segments);
    SegmentLength_t querySegmentLength() {return segmentLength;};
    SegmentLength_t querySegmentRemainLength() {return segmentRemainLength;};
    bool executeSegment(TimeStamp_t timeStamp);

    // Default constructor: create an empty segment
    Segment() {};
    Segment(SegmentLength_t segmentLength, SegmentIndex_t segmentIndex, 
            SegmentPreemption_t SegmentPreemption):
            segmentLength(segmentLength), segmentIndex(segmentIndex), segmentPreemption(segmentPreemption)
            {};
};

/**
 * @brief Virtual class on task, inherit to SSTask or DAGTask.
*/
class Task {

protected:
    ProcessorAffinity_t processorAffinity = CPU;
    bool processorMaskEnabled = false;
    std::vector<processor::ProcessorIndex_t> processorMasks = {};

    TaskPreemption_t taskPreemption = PREEMPTIVE;
    unsigned int segmentCount = 0;
    std::vector<Segment> segments = {};
    TimeStamp_t taskReleaseTime = 0;
    TimeStamp_t taskAbsoluteDeadline = 0;
    TaskRTProperty_t taskRealTimeProperty = TaskRTProperty_t::UNKNOWN;

public:
    unsigned int querySegmentCount() {return segmentCount;};
    bool isTaskCompleted();
    bool isSegmentReady(SegmentIndex_t segmentIndex) {return segments[segmentIndex].isSegmentReady();};
    /**
     * @brief Execute the given segment by 1 unit, need to take the preemption into account.
     * @return True if segment is succuessfully executed, otherwise false.
    */
    bool executeSegment(SegmentIndex_t segmentIndex);
    /**
     * @brief Create one new segment and inserted in the back
    */
    bool createNewSegment(SegmentLength_t segmentLength);

    // Default constructor: create an empty task
    Task() {};
    Task(TimeStamp_t taskReleaseTime, TimeStamp_t taskAbsoluteDeadline,
        TaskRTProperty_t taskRealTimeProperty, unsigned int segmentCount,
        ProcessorAffinity_t processorAffinity, TaskPreemption_t taskPreemption):
        taskReleaseTime(taskReleaseTime), taskAbsoluteDeadline(taskAbsoluteDeadline),
        taskRealTimeProperty(taskRealTimeProperty), segmentCount(segmentCount),
        processorAffinity(processorAffinity), taskPreemption(taskPreemption)
        {};

    void setProcessorMasks(std::vector<processor::ProcessorIndex_t> & processorMasks);
    bool isProcessorMaskEnabled() {return processorMaskEnabled;};
    bool isInsideProcessorMasks(processor::ProcessorIndex_t processorGlobalIndex);
};

class SSTask : public Task {

    SSTaskState_t SSTaskState;
    SegmentIndex_t currentSegmentIndex;

public:
    // Override the base class method, update the task state and current index.
    bool executeSegment(SegmentIndex_t SegmentIndex);
    SSTaskState_t querySSTaskState();
    Segment & getCurrentSegment() {return segments[currentSegmentIndex];};
};

class DAGTask : public Task {

    // TODO

};

#endif // task.h
