cmake_minimum_required(VERSION 3.15)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXTENSIONS OFF)

project(DGSEM LANGUAGES CXX)

add_library(dgsem INTERFACE)
target_include_directories(dgsem INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_subdirectory(examples)