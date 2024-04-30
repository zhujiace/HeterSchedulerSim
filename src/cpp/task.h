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
/**
 * @brief Describe a segment (SS Task Model) or node (DAG Task Model). 
*/
class Segment {
    SegmentPreemption_t segmentPreemption;
    SegmentLength_t segmentLength;
    SegmentLength_t segmentRemainLength;

    // The segment can be executed if and only if all the dependent segments are finished.
    unsigned int dependencyCount;
    std::vector<Segment *> dependentSegment;
public:
    bool isSegmentCompleted() {return segmentRemainLength==0;};
    bool isSegmentReady();
    SegmentLength_t querySegmentRemainLength();
};

class Task {

    ProcessorAffinity_t processorAffinity;
    TaskPreemption_t taskPreemption;


};
#endif // task.h
