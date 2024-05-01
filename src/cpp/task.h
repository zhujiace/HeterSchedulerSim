/*
Copy Right. The EHPCL Authors.
*/

#ifndef TASK_H
#define TASK_H

#include <vector>

#include "affinity.h"

enum TaskPreemption_t {
    PREEMPTIVE,
    NONPREEMPTIVE,
    UNKNOWN
};

typedef TaskPreemption_t SegmentPreemption_t;
typedef unsigned int SegmentLength_t;
typedef unsigned int SegmentIndex_t;


/**
 * @brief Describe a segment (SS Task Model) or node (DAG Task Model). 
*/
class Segment {
    SegmentPreemption_t segmentPreemption;
    SegmentLength_t segmentLength;
    SegmentLength_t segmentRemainLength;
    SegmentIndex_t segmentIndex;

    // The segment can be executed if and only if all the dependent segments are finished.
    unsigned int dependencyCount;
    // The dependent segments may come from other tasks, set to nullptr if none.
    std::vector<Segment *> dependentSegment;
public:
    bool isSegmentCompleted() {return segmentRemainLength==0;};
    bool isSegmentReady();
    void addToDependency(Segment & segment);
    bool configureDependency(std::vector<Segment *> segments);
    SegmentLength_t querySegmentLength() {return segmentLength;};
    SegmentLength_t querySegmentRemainLength() {return segmentRemainLength;};
    bool executeSegment();
};

/**
 * @brief Virtual class on task, inherit to SSTask or DAGTask.
*/
class Task {

protected:
    ProcessorAffinity_t processorAffinity;
    TaskPreemption_t taskPreemption;
    unsigned int segmentCount;
    std::vector<Segment> segments;

public:
    unsigned int querySegmentCount() {return segmentCount;};
    bool isTaskCompleted();
    bool isSegmentReady(SegmentIndex_t segmentIndex) {return segments[segmentIndex].isSegmentReady();};
    /**
     * @brief Execute the given segment by 1 unit, need to take the preemption into account.
     * @return True if segment is succuessfully executed, otherwise false.
    */
    bool executeSegment(SegmentIndex_t segmentIndex);

};

enum SSTaskState_t {
    EXECUTING,
    SUSPENSION,
    READY,
    UNKNOWN,
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

#endif // task.h
