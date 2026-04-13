cmake_minimum_required(VERSION 3.28)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ========================
# backend 开关
# ========================
option(ENABLE_CUDA "Enable CUDA backend" OFF)
option(ENABLE_OPENMP "Enable OpenMP backend" ON)
option(ENABLE_SERIAL "Enable Serial backend" OFF)

if(NOT ENABLE_CUDA AND NOT ENABLE_OPENMP AND NOT ENABLE_SERIAL)
    message(FATAL_ERROR "At least one backend must be enabled!")
endif()

# ========================
# ⚠️ CUDA 编译器必须在 project 前设置
# ========================
if(ENABLE_CUDA)
    set(CMAKE_C_COMPILER gcc-12)
    set(CMAKE_CXX_COMPILER g++-12)
    set(CMAKE_CUDA_HOST_COMPILER g++-12)
    set(CMAKE_CUDA_COMPILER nvcc)
    project(DGSEM LANGUAGES CXX CUDA)
else()
    project(DGSEM LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXTENSIONS OFF)

if(ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# ========================
# 清除旧 Kokkos 变量
# ========================
unset(Kokkos_DEVICES CACHE)
unset(Kokkos_OPTIONS CACHE)

# ========================
# Kokkos（强制 Fetch）
# ========================
include(FetchContent)

set(Kokkos_ENABLE_CUDA    ${ENABLE_CUDA}   CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_OPENMP  ${ENABLE_OPENMP} CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_SERIAL  ${ENABLE_SERIAL} CACHE BOOL "" FORCE)

if(ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda"
    )
    set(Kokkos_ARCH_AMPERE89 ON CACHE BOOL "" FORCE)
endif()

FetchContent_Declare(
    Kokkos
    URL https://github.com/kokkos/kokkos/archive/refs/tags/4.7.02.tar.gz
)
FetchContent_MakeAvailable(Kokkos)

# ========================
# backend info
# ========================
message(STATUS "Backend configuration:")
message(STATUS "  CUDA   : ${ENABLE_CUDA}")
message(STATUS "  OpenMP : ${ENABLE_OPENMP}")
message(STATUS "  Serial : ${ENABLE_SERIAL}")

# ========================
# VTK
# ========================
find_package(VTK REQUIRED COMPONENTS CommonCore CommonDataModel IOXML FiltersCore)

# ========================
# dgsem
# ========================
add_library(dgsem INTERFACE)

target_include_directories(dgsem
        INTERFACE
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

target_link_libraries(dgsem INTERFACE Kokkos::kokkos)
target_link_libraries(dgsem INTERFACE ${VTK_LIBRARIES})

if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    add_compile_options(
        --expt-relaxed-constexpr
        --expt-extended-lambda)
endif()

add_subdirectory(examples)