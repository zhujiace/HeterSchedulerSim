/*
Copy Right. The EHPCL Authors.
*/

#include <iostream>
#include <sstream>
#include "interface.h"

bool Interface::readCommands() {
    std::string line;
    std::cerr << ">>> ";
    while (std::getline(std::cin, line)) {
        std::cout << processCommand(line) << std::endl;
        if (quitFlag) return true;
        std::cerr << ">>> ";
    }

    return true;
}

std::string Interface::processCommand(const std::string & command) {
    if (command.empty()) return "Unknown command";
    std::istringstream ss(command);
    std::string cmd;
    ss >> cmd;

    std::string args = command.substr(cmd.length());

    auto it = command_map.find(cmd);
    if (it != command_map.end()) {
        return it->second(args);
    } else {
        return "Unknown command";
    }
}

Interface::Interface() {

    command_map = {
        {"queryCurrentTimeStamp", [this](const std::string&)
            {return std::to_string(getSimulator().queryCurrentTimeStamp());}},
        {"update", [this](const std::string&) 
            {getSimulator().updateProcessorAndTask();return "Updated";}},
        {"quit", [this](const std::string&)
            {quitFlag = true; return "Exiting...";}},
        {"printSimulatorState", [this](const std::string&)
            {getSimulator().printSimulatorStates();return "Simulator Status Printed";}},
    };
    command_map["createProcessor"] = 
        std::bind(&Interface::createProcessor, this, std::placeholders::_1);
    command_map["createHeterSSTask"] =
        std::bind(&Interface::createNewHeterSSTask, this, std::placeholders::_1);
}

ProcessorAffinity_t Interface::stringtoProcessorAffinity(const std::string & procAff) {
    for (unsigned int i = 0; i < 10; i++)
        if (procAff == processor::ProcessorTypeNames[i])
            return (ProcessorAffinity_t)i;
    return ProcessorAffinity_t::UNKNOWN;
}


std::string Interface::createProcessor(const std::string & args) {
    std::istringstream ss(args);
    std::string procType;
    ss >> procType; // first argument, processor type

    processor::ProcessorType_t processorAffinity = stringtoProcessorAffinity(procType);
    
    std::string procNum = "";
    int processorCount = 1;
    if (ss >> procNum) {
        processorCount = std::stoi(procNum);
    }
    if (simulator.createNewProcessors(processorAffinity, processorCount))
        return "Created successfully";
    
    return "Error Occurred";
}

std::string Interface::createNewHeterSSTask(const std::string & args) {
    std::istringstream ss(args);
    int procNum = 1;
    std::string temp;
    ss >> temp;
    procNum = std::stoi(temp);

    std::vector<ProcessorAffinity_t> processorTypes = {};
    for (unsigned int i = 0; i < procNum; i++) {
        ss >> temp;
        processorTypes.push_back(stringtoProcessorAffinity(temp));
    }
    std::vector<unsigned int> segments = {};
    while (ss >> temp) {
        unsigned int segLength = std::stoi(temp);
        segments.push_back(segLength);
    }
    simulator.createNewHeterSSTaskWithVector(processorTypes, segments);
    return "Created successfully";
}
