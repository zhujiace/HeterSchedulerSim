/*
Copy Right. The EHPCL Authors.
*/

#ifndef INTERFACE_H
#define INTERFACE_H

#include <unordered_map>
#include <functional>
#include "simulator.h"

class Interface {

    Simulator simulator;

    std::string tempOutput;

    std::unordered_map<std::string, std::function<std::string(const std::string&)>>
        command_map;

    bool quitFlag = false;

public:

    bool readCommands();

    std::string processCommand(const std::string & command);

    Simulator & getSimulator() {return simulator;};

    Interface();

    ProcessorAffinity_t stringtoProcessorAffinity(const std::string & processorAffinity);

    std::string createProcessor(const std::string & args);

    std::string createNewHeterSSTask(const std::string & args);

    std::string setSimulationTimeBound(const std::string & args);

    std::string queryProcessorStates();

    std::string queryProcessorState(const std::string & args);

    std::string segmentStateHelperFunc(unsigned int taskId, unsigned int segId);

    std::string queryTaskSpecifiedSegmentState(const std::string & args);

    std::string queryTaskSegmentStates(const std::string & args);

    std::string queryTaskStates();

    std::string scheduleSegmentOnProcessor(const std::string & args);

    std::string updateProcessorAndTask();

};

#endif // interface.h