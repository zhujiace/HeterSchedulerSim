cmake_minimum_required(VERSION 3.10)
project(HeterSchedulerSim)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DEFF_DEBUG")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DDEMO -DPERF_DEBUG")
add_executable(main src/cpp/processor.cpp src/cpp/scheduler.cpp src/cpp/simulator.cpp src/cpp/task.cpp test/main.cpp)
target_link_libraries(  main )

# add_executable(main_2 test_single.cpp)
# target_link_libraries(  main_2  ${OURPATH}/libopenvino.so ${OpenCV_LIBS} )