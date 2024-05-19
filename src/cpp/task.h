/*
Copy Right. The EHPCL Authors.
*/

#ifndef TASK_H
#define TASK_H

#include <vector>
#include <queue>

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
    NONERT, // Current simulator does not support none RT Task.
};

typedef TaskPreemption_t SegmentPreemption_t;
typedef unsigned int SegmentLength_t;
typedef unsigned int SegmentIndex_t;

typedef unsigned int TaskIndex_t;

enum SegmentState_t {
    SEG_EXECUTING,
    SEG_READY,
    SEG_NOTREADY,
    SEG_FINISHED,
    SEG_UNKNWON
};

enum SSTaskState_t {
    SS_EXECUTING,
    SS_SUSPENSION,
    SS_READY,
};

enum TaskState_t {
    TASKS_EXECUTING, // deprecated, tasks can be inter-parallel
    TASKS_READY,
    TASKS_FINISHED,
    TASKS_MISSDDL,
    TASKS_UNKNOWN,
};

/*enum HeterSSTaskState_t {
    TASKS_EXECUTING,
    TASKS_FINISHED,
    TASKS_READY,
    TASKS_MISSDDL,
    TASKS_UNKNOWN,
};*/

typedef unsigned long long TimeStamp_t;

typedef unsigned char TaskRTPriority_t;

// The value is same as "SCHED_***" defined in <sched.h>
enum TaskRTSchedulePolicy_t {
    HETER_SCHED_OTHER,
    HETER_SCHED_FIFO,
    HETER_SCHED_RR,
    HETER_SCHED_BATCH,
    HETER_SCHED_ISO,
    HETER_SCHED_IDLE,
    HETER_SCHED_DEADLINE,
};

};

using namespace task;


class Processor;

/**
 * @brief Describe a segment (SS Task Model) or node (DAG Task Model). 
*/
class Segment {
    SegmentPreemption_t segmentPreemption = SegmentPreemption_t::PREEMPTIVE;
    SegmentLength_t segmentLength = 0;
    SegmentLength_t segmentRemainLength = 0;
    // The segment index inside its task.
    SegmentIndex_t segmentIndex = 0;

    ProcessorAffinity_t segmentAffinity;
    Processor * currentProcessor;

    // The segment can be executed if and only if all the precedent segments are finished.
    // The dependent segments may come from other tasks, set to nullptr if none.
    std::vector<Segment *> precedingSegments;
    std::vector<Segment *> succeedingSegments;

    std::vector<TimeStamp_t> executedAt = {};

    // Local storage to speed up the system.
    // Set to true if the corresponding methods return true.
    bool segmentCompleted = false;
    bool segmentReady = false;
public:
    bool isSegmentCompleted() {return segmentRemainLength==0;};
    void markSegmentReady() {segmentReady = true;};
    bool isSegmentReady();
    /**
     * @brief Add the given segment to the precedingSegments
     * @param segment reference to the prec segment
    */
    void addToDependency(Segment & segment);

    SegmentLength_t querySegmentLength() {return segmentLength;};
    SegmentLength_t querySegmentRemainLength() {return segmentRemainLength;};
    ProcessorAffinity_t querySegmentProcessorAffinity() {return segmentAffinity;}
    // may return false if the non-preemptive segment is not executed continuously
    bool executeSegment(TimeStamp_t timeStamp);

    // Default constructor: create an empty segment
    Segment() {};
    Segment(SegmentLength_t segmentLength, ProcessorAffinity_t processorAffinity):
            segmentLength(segmentLength), segmentAffinity(processorAffinity)
            {};
    Segment(SegmentLength_t segmentLength, ProcessorAffinity_t processorAffinity, 
            SegmentPreemption_t SegmentPreemption):
            segmentLength(segmentLength), segmentAffinity(processorAffinity),
            segmentPreemption(segmentPreemption) {};
    
    /**
     * @brief Task may be period, the segment is reinited and executed again.
     * @return True if reseted successfully, False if the reset behavior is inproper,
     * e.g. reset before the segment has finished.
    */
    bool resetSegment();

    void setCurrentProcessor(Processor * processor) {currentProcessor = processor;};
    Processor * getCurrentProcessor() { return currentProcessor;};
};

/**
 * @brief Class on task, can inherit to SSTask or DAGTask.
*/
class Task {

protected:

    // Internal storage, the instances of segments
    std::vector<Segment> segments = {};

    // Internal fixed properties
    ProcessorAffinity_t processorAffinity = CPU;
    TaskRTProperty_t taskRealTimeProperty = TaskRTProperty_t::HARDRT;

    // TODO: Future feature, to support parallel execution of 
    // segments inside this task.
    bool ifParallel = false;

    // Scheduling related properties
    TimeStamp_t taskRelativeDeadline = 0;
    TimeStamp_t taskAbsoluteDeadline = 0;
    TimeStamp_t taskExecutionTime = 0;

    TimeStamp_t taskPeriod = 0;
    TaskRTPriority_t taskPriority = 99;
    TaskRTSchedulePolicy_t taskSchedulePolicy = HETER_SCHED_FIFO;

    // Runtime maintainance (storage)
    TaskState_t taskState = TaskState_t::TASKS_UNKNOWN;
   
    bool processorMaskEnabled = false;
    std::vector<unsigned int> processorMasks = {};

    bool taskCompleted = false;

    std::vector<SegmentState_t> segmentStates;
    std::vector<Segment *> readySegments = {};

public:

    Segment & createNewSegment(ProcessorAffinity_t processorAffinity, SegmentLength_t segmentLength);

    /**
     * @brief Initialize the internal task vector by a given vector.
     * @return False if the format is not correct, otherwise true.
     * @example Input segments: Type = {CPU,GPU}, segments = {3,1,2,4,5} -> C-G-C-G-C.
     * @attention Current version requires the processorTypes to be different.
    */
    bool initializeTaskByVector(std::vector<ProcessorAffinity_t> & processorType,
                                std::vector<SegmentLength_t> & segments);

    TimeStamp_t queryTaskPeriod() {return taskPeriod;}
    TimeStamp_t queryTaskRelativeDeadline() {return taskRelativeDeadline;}
    TaskRTPriority_t queryTaskRTPriority() {return taskPriority;}
    void setTaskPeriod(TimeStamp_t taskPeriod) {this->taskPeriod = taskPeriod;};
    void setTaskRelativeDeadline(TimeStamp_t deadline) {this->taskRelativeDeadline = deadline;};
    void setTaskRTPriority(TaskRTPriority_t priority) {this->taskPriority = priority;};

    double queryTaskUtilization();
    double querySingleTaskUtilization(ProcessorAffinity_t processorAffinity);

    bool releaseTask(TimeStamp_t currentTime);
    
    TaskState_t checkTaskStates();
    TaskState_t queryTaskState() {return taskState;}
    void setTaskState(TaskState_t state) {taskState = state;}
    
    bool isAllSegmentsCompleted();
    bool isTaskCompleted() {return isAllSegmentsCompleted();}
    
    // true if miss
    bool checkWhetherMissDDL(TimeStamp_t currentTime);

    // TODO: improve task status printing
    friend std::ostream & operator<<(std::ostream & os, const Task & task) {
        os << std::string("State: ");
        switch (task.taskState) {
            case TASKS_EXECUTING:
                os << std::string("executing");break;
            case TASKS_FINISHED:
                os << std::string("finished");break;
            case TASKS_READY:
                os << std::string("ready");break;
            case TASKS_MISSDDL:
                os << std::string("missddl");break;
            default:
                os << std::string("unknown");break;
        }
        return os;
    }

    unsigned int querySegmentCount() {return segments.size();};
    SegmentLength_t querySegmentExecutionTime();

    bool isSegmentReady(SegmentIndex_t segmentIndex) {return segments[segmentIndex].isSegmentReady();};
    /**
     * @brief Execute the given segment by 1 unit, need to take the preemption into account.
     * @return True if segment is succuessfully executed, otherwise false.
    */
    bool executeSegment(SegmentIndex_t segmentIndex, TimeStamp_t timeStamp);
    bool executeFirstReadySegment(TimeStamp_t timeStamp);

    bool _resetAllSegments();
    bool resetTask() {return _resetAllSegments();};

    // Default constructor: create an empty task
    Task() {segments.reserve(20);};
    Task(TaskRTPriority_t taskPriority, TimeStamp_t taskPeriod):
    taskPriority(taskPriority), taskPeriod(taskPeriod), taskRealTimeProperty(HARDRT)
    {segments.reserve(20);};

    void setProcessorMasks(std::vector<unsigned int> & processorMasks)
        {this->processorMasks = processorMasks;};
    bool isProcessorMaskEnabled() {return processorMaskEnabled;};
    bool isInsideProcessorMasks(unsigned int processorGlobalIndex);

    TaskRTProperty_t queryTaskRTProperty() {return taskRealTimeProperty;};

    bool setTaskScheduled();
    // consider changes on heter ss task
    bool setTaskPreempted();
    void setFirstSegmentReady() {segments[0].markSegmentReady();}

    /**
     * @brief Configure dependency: seg2 depends on seg1
    */
    void setSegmentDependency(SegmentIndex_t seg1, SegmentIndex_t seg2);
    std::vector<Segment *> & getReadySegments();
    std::vector<Segment *> & queryReadySegments() {return readySegments;}
    Segment * getFirstReadySegment(ProcessorAffinity_t processorAffinity);
    Segment & getSegment(SegmentIndex_t segmentIndex) {
        return segments[segmentIndex];
    }
};

class SSTask : public Task {

    SSTaskState_t SSTaskState;
    SegmentIndex_t currentSegmentIndex;

public:
    // Override the base class method, update the task state and current index.
    bool executeSegment(SegmentIndex_t SegmentIndex) {return true;};
    SSTaskState_t querySSTaskState() {return SSTaskState;};
    Segment & getCurrentSegment() {return segments[currentSegmentIndex];};
};

class DAGTask : public Task {

    // TODO

};

#endif // task.h
