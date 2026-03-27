cmake_minimum_required(VERSION 3.23)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ========================
# 用户可选 backend 开关
# ========================
option(ENABLE_CUDA "Enable CUDA backend" OFF)
option(ENABLE_OPENMP "Enable OpenMP backend" ON)
option(ENABLE_SERIAL "Enable Serial backend" ON)

if(NOT ENABLE_CUDA AND NOT ENABLE_OPENMP AND NOT ENABLE_SERIAL)
    message(FATAL_ERROR "At least one backend must be enabled!")
endif()

# ========================
# project（必须在语言确定后）
# ========================
if(ENABLE_CUDA)
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

    add_compile_options(
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
    )
endif()

# ========================
# Kokkos
# ========================
find_package(Kokkos CONFIG QUIET)

if(NOT Kokkos_FOUND)
    message(STATUS "Kokkos not found. Fetching...")

    include(FetchContent)

    set(Kokkos_ENABLE_CUDA ${ENABLE_CUDA} CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_OPENMP ${ENABLE_OPENMP} CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_SERIAL ${ENABLE_SERIAL} CACHE BOOL "" FORCE)

    if(ENABLE_CUDA)
        # 可由 -D 覆盖
        set(Kokkos_ARCH_AMPERE86 ON CACHE BOOL "" FORCE)
    endif()

    FetchContent_Declare(
        Kokkos
        URL https://github.com/kokkos/kokkos/archive/refs/tags/4.4.01.tar.gz
    )
    FetchContent_MakeAvailable(Kokkos)
endif()

# ========================
# 打印 backend 信息
# ========================
message(STATUS "Backend configuration:")
message(STATUS "  CUDA   : ${ENABLE_CUDA}")
message(STATUS "  OpenMP : ${ENABLE_OPENMP}")
message(STATUS "  Serial : ${ENABLE_SERIAL}")

# ========================
# 你的库
# ========================
add_library(dgsem INTERFACE)

target_include_directories(dgsem
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

target_link_libraries(dgsem INTERFACE Kokkos::kokkos)

add_subdirectory(examples)