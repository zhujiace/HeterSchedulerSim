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

typedef unsigned int HeterTaskIndex_t;

enum SegmentState_t {
    SEG_READY,
    SEG_NOTREADY,
    SEG_FINISHED,
    UNKNWON
};

enum SSTaskState_t {
    SS_EXECUTING,
    SS_SUSPENSION,
    SS_READY,
};

enum HeterSSTaskState_t {
    TASKS_EXECUTING,
    TASKS_FINISHED,
    TASKS_READY,
    TASKS_MISSDDL,
    TASKS_UNKNOWN,
};

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

/**
 * @brief Describe a segment (SS Task Model) or node (DAG Task Model). 
*/
class Segment {
    SegmentPreemption_t segmentPreemption = SegmentPreemption_t::PREEMPTIVE;
    SegmentLength_t segmentLength = 0;
    SegmentLength_t segmentRemainLength = 0;
    // The segment index inside its task.
    SegmentIndex_t segmentIndex = 0;

    // The segment can be executed if and only if all the dependent segments are finished.
    // The dependent segments may come from other tasks, set to nullptr if none.
    std::vector<Segment *> dependentSegments;
    std::vector<Segment *> dependentedBySegments;

    std::vector<TimeStamp_t> executedAt = {};

    // Local storage to speed up the system.
    // Set to true if the corresponding methods return true.
    bool segmentCompleted = false;
    bool segmentReady = false;
public:
    bool isSegmentCompleted() {return segmentRemainLength==0;};
    void markSegmentReady() {segmentReady = true;};
    bool isSegmentReady();
    void addToDependency(Segment & segment);
    void configureDependency(std::vector<Segment *> & segments)
        {dependentSegments = segments;}
    SegmentLength_t querySegmentLength() {return segmentLength;};
    SegmentLength_t querySegmentRemainLength() {return segmentRemainLength;};
    bool executeSegment(TimeStamp_t timeStamp);

    // Default constructor: create an empty segment
    Segment() {};
    Segment(SegmentLength_t segmentLength, SegmentIndex_t segmentIndex, 
            SegmentPreemption_t SegmentPreemption):
            segmentLength(segmentLength), segmentIndex(segmentIndex), segmentPreemption(segmentPreemption)
            {};
    
    /**
     * @brief Task may be period, the segment is reinited and executed again.
     * @return True if reseted successfully, False if the reset behavior is inproper,
     * e.g. reset before the segment has finished.
    */
    bool resetSegment();
};


class HeterSSTask;

/**
 * @brief Class on task, can inherit to SSTask or DAGTask.
*/
class Task {

protected:
    ProcessorAffinity_t processorAffinity = CPU;
    bool processorMaskEnabled = false;
    std::vector<unsigned int> processorMasks = {};

    TaskPreemption_t taskPreemption = TaskPreemption_t::PREEMPTIVE;
    std::vector<Segment> segments = {};
    TaskRTProperty_t taskRealTimeProperty = TaskRTProperty_t::HARDRT;

    bool taskCompleted = false;

    std::vector<SegmentState_t> segmentStates;
    std::vector<Segment *> readySegments = {};

    // TODO: Future feature, to support parallel execution of 
    // segments inside this task.
    // Argue: In this case, we should split the task
    bool ifParallel = false;

    HeterSSTask * belongedHeterSSTask = nullptr;

public:
    unsigned int querySegmentCount() {return segments.size();};
    SegmentLength_t querySegmentExecutionTime();

    bool isTaskCompleted();
    bool isSegmentReady(SegmentIndex_t segmentIndex) {return segments[segmentIndex].isSegmentReady();};
    /**
     * @brief Execute the given segment by 1 unit, need to take the preemption into account.
     * @return True if segment is succuessfully executed, otherwise false.
    */
    bool executeSegment(SegmentIndex_t segmentIndex, TimeStamp_t timeStamp);
    bool executeFirstReadySegment(TimeStamp_t timeStamp);
    /**
     * @brief Create one new segment and inserted in the back
    */
    bool createNewSegment(SegmentLength_t segmentLength);

    bool _resetAllSegments();
    bool resetTask() {return _resetAllSegments();};

    // Default constructor: create an empty task
    Task() {};
    Task(TaskRTProperty_t taskRealTimeProperty,
         ProcessorAffinity_t processorAffinity, TaskPreemption_t taskPreemption):
         taskRealTimeProperty(taskRealTimeProperty),
         processorAffinity(processorAffinity), taskPreemption(taskPreemption)
         {};

    void setProcessorMasks(std::vector<unsigned int> & processorMasks)
        {this->processorMasks = processorMasks;};
    bool isProcessorMaskEnabled() {return processorMaskEnabled;};
    bool isInsideProcessorMasks(unsigned int processorGlobalIndex);
    ProcessorAffinity_t queryProcessorAffinity() {return processorAffinity;};
    TaskPreemption_t queryTaskPreemption() {return taskPreemption;};
    // Search all the tasks
    SegmentState_t queryTaskState();

    TaskRTProperty_t queryTaskRTProperty() {return taskRealTimeProperty;};

    bool setTaskScheduled();
    // consider changes on heter ss task
    bool setTaskPreempted();
    void setFirstSegmentReady() {segments[0].markSegmentReady();}
    void setBelongedHeterSSTaskset(HeterSSTask * htask) {belongedHeterSSTask = htask;}
    HeterSSTask * getBelongHeterSSTaskset() {return belongedHeterSSTask;}

    /**
     * @brief Configure dependency: seg2 depends on seg1
    */
    void setSegmentDependency(SegmentIndex_t seg1, SegmentIndex_t seg2);
    std::vector<Segment *> getReadySegments();
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

class Processor;

/**
 * @brief Represent one task (tau_i) in heterogenous computing platform, 
 *  using the self-suspension model.
*/
class HeterSSTask {


    /**
     * @brief Different subtasks which combines to make the whole task, 
     * executed in an interleaved pattern.
     * @example C0-G0-C1-G1-C2-G2-C3, where Ci are CPU Task, Gi are GPU Task,
     * each Ci or Gi is a Segment itself.
    */
    std::vector<Task> internalTasks = {};

    unsigned int processorTypeCount = 0;

    TimeStamp_t taskRelativeDeadline = 0;
    TimeStamp_t taskAbsoluteDeadline = 0;
    TimeStamp_t taskExecutionTime = 0;

    TimeStamp_t taskPeriod = 0;

    HeterSSTaskState_t heterSSTaskState = HeterSSTaskState_t::TASKS_UNKNOWN;

    TaskRTPriority_t heterSSTaskPriority = 99;
    TaskRTSchedulePolicy_t heterSSTaskSchedulePolicy = HETER_SCHED_FIFO;

    Processor * currentProcessor = nullptr;

public:


    Task & _createNewTask(TaskRTProperty_t taskRealTimeProperty, 
                        ProcessorAffinity_t processorAffinity, TaskPreemption_t taskPreemption);
    Task & createNewRTTask(ProcessorAffinity_t processorAffinity, TaskPreemption_t taskPreemption);

    Task & getTask(ProcessorAffinity_t processorAffinity);
    Task & getReadyTask();
    // TODO: Support DAG tasks where multiple subtasks are ready simultaneously.
    std::vector<Task *> getReadyTasks() {return {};};
    bool createNewSegmentForTask(ProcessorAffinity_t processorAffinity, SegmentLength_t segmentLength);

    /**
     * @brief Initialize the internal task vector by a given vector.
     * @return False if the format is not correct, otherwise true.
     * @example Input segments: Type = {CPU,GPU}, segments = {3,1,2,4,5} -> C-G-C-G-C.
     * @attention Current version requires the processorTypes to be different.
    */
    bool initializeTaskByVector(std::vector<ProcessorAffinity_t> processorType, std::vector<SegmentLength_t> segments);

    TimeStamp_t queryTaskPeriod() {return taskPeriod;}
    TimeStamp_t queryTaskRelativeDeadline() {return taskRelativeDeadline;}
    TaskRTPriority_t queryTaskRTPriority() {return heterSSTaskPriority;}
    void setTaskPeriod(TimeStamp_t taskPeriod) {this->taskPeriod = taskPeriod;};
    void setTaskRelativeDeadline(TimeStamp_t deadline) {this->taskRelativeDeadline = deadline;};
    void setTaskRTPriority(TaskRTPriority_t priority) {this->heterSSTaskPriority = priority;};

    double queryTaskUtilization();
    double querySingleTaskUtilization(ProcessorAffinity_t processorAffinity);

    bool releaseTask(TimeStamp_t currentTime);
    // Enumerate over all the subtasks
    void checkHeterSSTaskFinishOrReady();
    HeterSSTaskState_t queryHeterSSTaskState();
    void setHeterSSTaskState(HeterSSTaskState_t hstate) {heterSSTaskState = hstate;}
    bool isAllTasksCompleted();
    // true if miss
    bool checkWhetherMissDDL(TimeStamp_t currentTime);

    friend std::ostream & operator<<(std::ostream & os, const HeterSSTask & task) {
        os << std::string("State: ");
        switch (task.heterSSTaskState) {
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

    HeterSSTask() {}
    HeterSSTask(const HeterSSTask & otherHeterTask);
};

#endif // task.h
