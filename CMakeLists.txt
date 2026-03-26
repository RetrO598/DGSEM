cmake_minimum_required(VERSION 3.15)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXTENSIONS OFF)

project(DGSEM LANGUAGES CXX)

find_package(OpenMP REQUIRED)

find_package(Kokkos CONFIG QUIET)

if(Kokkos_FOUND)
    message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
else()
    message(STATUS "Kokkos not found externally. Fetching via FetchContent.")

    include(FetchContent)

    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
    set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)

    FetchContent_Declare(
        Kokkos
        URL https://github.com/kokkos/kokkos/archive/refs/tags/4.4.01.tar.gz
    )
    FetchContent_MakeAvailable(Kokkos)
endif()


add_library(dgsem INTERFACE)
target_include_directories(dgsem
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_link_libraries(dgsem INTERFACE Kokkos::kokkos)

add_subdirectory(examples)