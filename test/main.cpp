#include <iostream>
#include <string>
#include <fstream>

#include "scheduler.h"

#define TASK_NUM 5
#define SEGMENT_NUM 5

int main () {
    
    Scheduler scheduler;
    scheduler.startScheduleLoop();

    std::cout << "This is a test script for the simulator." << std::endl;
    return 0;
}