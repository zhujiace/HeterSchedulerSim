/*
Copy Right. The EHPCL Authors.
*/

#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "processor.h"

class Scheduler {

public:
    bool schedule(Task task);

};

/**
 * @brief The simulator single-instance exposed to users.
 * 
*/
class Simulator {

    std::vector<Processor> processors = {};

    std::vector<Task> tasks = {};

public:

    /**
     * @brief Create one new processor and insert in the vector end,
     * set the temperate global index as vector index.
     * @return True if successfully created.
    */
    bool createNewProcessor(processor::ProcessorType_t processorType);
    bool sortProcessorsByType();

    /**
     * @brief Create one new task and insert in the vector end,
     * without initilize the dependencies.
     * @return True if successfully create.
    */
    bool createNewTask();

};

#endif // scheduler.h
