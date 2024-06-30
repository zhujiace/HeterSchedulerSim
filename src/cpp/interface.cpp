/*
Copy Right. The EHPCL Authors.
*/

#include <iostream>
#include <sstream>
#include "interface.h"

bool Interface::readCommands() {
    std::string line;
    if (interactive) std::cerr << ">>> ";
    while (std::getline(std::cin, line)) {
        std::cout << processCommand(line) << std::endl;
        if (quitFlag) return true;
        if (interactive) std::cerr << ">>> ";
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
        {"quit", [this](const std::string&)
            {quitFlag = true; return "Exiting...";}},
        {"printSimulatorState", [this](const std::string&)
            {getSimulator().printSimulatorStates();return "Simulator Status Printed";}},
        {"sortProcessors", [this](const std::string&)
            {getSimulator().sortProcessorsByType(); return "Sorted";}},
        {"isSimulationCompleted", [this](const std::string &)
            {return getSimulator().isSimulationCompleted()?"Yes":"";}},
        {"doesTaskMissDeadline", [this](const std::string &)
            {return getSimulator().doesTaskMissDeadline()?"Yes":"";}},
        {"updateProcessorAndTask", [this](const std::string &)
            {return updateProcessorAndTask();}},
        {"queryProcessorStates", [this](const std::string &)
            {return queryProcessorStates();}},
        {"queryTaskExecutionStates", [this](const std::string &)
            {return queryTaskExecutionStates();}},
        {"startSimulation", [this](const std::string &)
            {getSimulator().checkTaskRelease(); return "Initial Tasks Released";}},
    };
    command_map["createProcessor"] = 
        std::bind(&Interface::createProcessor, this, std::placeholders::_1);
    command_map["createHeterSSTask"] =
        std::bind(&Interface::createNewHeterSSTask, this, std::placeholders::_1);
    command_map["setSimulationTimeBound"] =
        std::bind(&Interface::setSimulationTimeBound, this, std::placeholders::_1);
    command_map["queryTaskState"] = 
        std::bind(&Interface::queryTaskState, this, std::placeholders::_1);
    command_map["scheduleSegmentOnProcessor"] =
        std::bind(&Interface::scheduleSegmentOnProcessor, this, std::placeholders::_1);
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
    std::string temp;
    ss >> temp;
    int period = 99;
    period = std::stoi(temp);
    ss >> temp;
    int procNum = 1;
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
    Task & task = simulator.createNewHeterSSTaskWithVector(processorTypes, segments);
    task.setTaskPeriod(period);
    task.setTaskRelativeDeadline(period);
    return "Created successfully";
}

std::string Interface::setSimulationTimeBound(const std::string & args) {
    unsigned int timeBound = std::stoi(args);
    simulator.setSimulationTimeBound(timeBound);
    return "Set bound to " + std::to_string(timeBound);
}

std::string Interface::queryProcessorStates() {
    std::string temp = "";
    for (unsigned int i = 0; i < simulator.queryProcessorCount(); i++) {
        temp += queryProcessorState(std::to_string(i));
        temp += " ";
    }
    return temp;
}


/**
 * @brief return the state of the specified processor
 * @param args "<procId>"
 * @return "<procType> <processorState>", 0-IDLE, 1-PREEMPTIVE, 2-NONPREEMPTIVE
*/
std::string Interface::queryProcessorState(const std::string & args) {
    unsigned int processorId = std::stoi(args);
    std::string res = "";
    res += std::to_string((int)simulator.getProcessor(processorId).queryProcessorType());
    res += " ";
    res += std::to_string((int)simulator.getProcessor(processorId).queryProcessorState());
    return res;
}

/**
 * @return "<affinity> <currentProcessor> <isSegmentReady> <length> <remainLength> "
 */
std::string Interface::segmentStateHelperFunc(unsigned int taskId, unsigned int segmentId) {
    std::string result = "";
    result += std::to_string(int(simulator.getTask(taskId).getSegment(segmentId).querySegmentProcessorAffinity()));
    result += " ";
    result += std::to_string(simulator.getTask(taskId).getSegment(segmentId).getCurrentProcessorIndex());
    result += " ";
    result += std::to_string(int(simulator.getTask(taskId).isSegmentReady(segmentId)));
    result += " ";
    result += std::to_string(simulator.getTask(taskId).getSegment(segmentId).querySegmentLength());
    result += " ";
    result += std::to_string(simulator.getTask(taskId).getSegment(segmentId).querySegmentRemainLength());
    result += " ";
    return result;
}

/**
 * @brief return current state of a specified segment
 * @param args "<taskId> <segmentId>"
 * @return "<affinity> <currentProcessor> <isSegmentReady> <length> <remainLength> "
*/
std::string Interface::queryTaskSpecifiedSegmentState(const std::string & args) {
    std::istringstream ss(args);
    std::string temp;
    ss >> temp;
    unsigned int taskId = std::stoi(temp);
    ss >> temp;
    unsigned int segmentId = std::stoi(temp);

    return segmentStateHelperFunc(taskId, segmentId);
}

/**
 * @brief return the states of all segments in this task
 * @param args "<taskId>"
 * @see queryTaskSpecifiedSegmentState
*/
std::string Interface::queryTaskSegmentStates(const std::string & args) {
    std::istringstream ss(args);
    std::string temp;
    ss >> temp;
    unsigned int taskId = std::stoi(temp);
    std::string result;
    for (unsigned int i = 0 ; i < simulator.getTask(taskId).querySegmentCount(); i++) {
        result += segmentStateHelperFunc(taskId, i);
    }
    return result;
}

/**
 * @param args "<procId> <taskId> <segmentId>"
 * @brief Send the segment schedlue command
*/
std::string Interface::scheduleSegmentOnProcessor(const std::string & args) {
    std::istringstream ss(args);
    std::string temp;
    ss >> temp;
    unsigned int procId = std::stoi(temp);
    ss >> temp;
    unsigned int taskId = std::stoi(temp);
    ss >> temp;
    unsigned int segmentId = std::stoi(temp);

    Task & task = simulator.getTask(taskId);
    Segment * segment = &(simulator.getTask(taskId).getSegment(segmentId));
    TimeStamp_t currentTime = simulator.queryCurrentTimeStamp();

    if (simulator.getProcessor(procId).scheduleTaskSpecifiedSegment(task, segment, currentTime)) return "Scheduled";
    else return "Scheule Error!";

    return "Unknown Error";
}

std::string Interface::updateProcessorAndTask() {
    int res = simulator.updateProcessorAndTask();
    if (simulator.doesTaskMissDeadline())
        std::cerr << "Task miss deadline! Please Exit!\n";
    return std::to_string(res) + " executed. Updated to timestamp " +
           std::to_string(simulator.queryCurrentTimeStamp());
}

/**
 * @return <t1.executedLength> <t2.executedLength> ...
 */
std::string Interface::queryTaskExecutionStates() {
    std::string result = "";
    for (unsigned int i = 0; i < simulator.queryTaskCount(); i++)
        result += (std::to_string(simulator.getTask(i).queryExecutedSegLength()) + " ");
    return result;
}

/**
 * @brief return the overall state of a task
 * @return <period> <s0.segmentState> <s1.segmentState> ...
 * @param args the task index
 * @note segmentState: <affinity> <currentProcessor> <isSegmentReady> <length> <remainLength> 
 */
std::string Interface::queryTaskState(const std::string & args) {
    std::istringstream ss(args);
    std::string temp;
    ss >> temp;
    unsigned int taskId = std::stoi(temp);
    std::string result = "";
    result += std::to_string(simulator.getTask(taskId).queryTaskPeriod());
    result += " ";
    result += queryTaskSegmentStates(args);
    return result;
}
