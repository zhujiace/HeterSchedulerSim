cmake_minimum_required(VERSION 3.10)
project(HeterSchedulerSim)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
file(GLOB SOURCE_FILES ./src/cpp/*.cpp)
include_directories(${CMAKE_SOURCE_DIR}/src/cpp)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DEFF_DEBUG")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DDEMO -DPERF_DEBUG")
add_executable(main ${SOURCE_FILES} test/main.cpp)
target_link_libraries(  main )

# add_executable(main_2 test_single.cpp)
# target_link_libraries(  main_2  ${OURPATH}/libopenvino.so ${OpenCV_LIBS} )3

# cmake -S . -B build