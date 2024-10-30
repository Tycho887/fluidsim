cmake_minimum_required(VERSION 3.10)
project(my_cpp_project)

set(CMAKE_CXX_STANDARD 17)
set(SOURCE_FILES src/main.cpp)

add_executable(my_program ${SOURCE_FILES})