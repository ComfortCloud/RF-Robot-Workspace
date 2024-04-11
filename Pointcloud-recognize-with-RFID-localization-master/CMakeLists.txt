cmake_minimum_required(VERSION 2.8.3)
project(LMcalculate)


find_package(Eigen3 REQUIRED)

include_directories(
	include
)

add_executable(LMcalculate src/preprocessing.cpp src/main.cpp)
add_executable(LMcalculate_cp src/preprocessing_cp.cpp src/main_cp.cpp)