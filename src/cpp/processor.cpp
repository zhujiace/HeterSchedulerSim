/*
Copy Right. The EHPCL Authors.
*/

#include "processor.h"

bool Processor::scheduleTask(Task & taskToSchedule, task::TimeStamp_t timeStamp) {

    if (&(taskToSchedule) == currentTask) return true;
    if (currentTask) {
        // There's task going on the processor.
        // Check whether both processor and task support preemption
        if (processorPreemption!=ProcessorPreemption_t::PREEMPTIVE) return false;
        if (currentTask->queryTaskPreemption()!=TaskPreemption_t::PREEMPTIVE) return false;
        currentTask->setTaskPreempted();
    }

    currentTask = &(taskToSchedule);
    currentTaskPriority = currentTask->getBelongHeterSSTaskset()->queryTaskRTPriority();
    taskToSchedule.setTaskScheduled();
    processorState = BUSY_NONPREEMPTIVE;
    if (processorPreemption==ProcessorPreemption_t::PREEMPTIVE)
        if (currentTask->queryTaskPreemption()==TaskPreemption_t::PREEMPTIVE)
            processorState = BUSY_PREEMPTIVE;
    return true;
}

bool Processor::workProcessor(task::TimeStamp_t timeStamp) {
    if (processorState == IDLE) return true;
    if (!currentTask->executeFirstReadySegment(timeStamp)) return false;

    // TODO: Add temporal records here

    if (currentTask->getReadySegments().size()==0) {
        currentTask->getBelongHeterSSTaskset()->checkHeterSSTaskFinishOrReady();
        processorState = IDLE;
        currentTask = nullptr;
    }
    return true;
}
