#include <iostream>
#include <string>
#include <fstream>

#include "scheduler.h"

#define TASK_NUM 5
#define SEGMENT_NUM 5

int main () {
    
    // Support schedule from processor side
    // Segments - SegmentGroup - Tasks

    std::cout << "Start Testing...\n";
    Scheduler scheduler;
    scheduler.initializeSimulation();
    bool res = scheduler.startScheduleLoop();

    std::cout << "Schedule result: " << res << std::endl;
    return 0;
}