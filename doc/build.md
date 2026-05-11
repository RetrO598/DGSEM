# Build Guide

DGSEM is configured with CMake and uses Kokkos for performance portability.
The current build system fetches Kokkos automatically through CMake.

## Requirements

- C++20 compiler.
- CMake 3.28 or newer.
- VTK development package.
- CUDA 12.1 or newer for CUDA builds.

## CPU/OpenMP Build

```bash
mkdir -p build
cd build
cmake .. -DENABLE_CUDA=OFF -DENABLE_OPENMP=ON -DENABLE_SERIAL=ON
make -j$(nproc)
```

## CUDA Build

```bash
mkdir -p build-cuda
cd build-cuda
cmake .. -DENABLE_CUDA=ON -DENABLE_OPENMP=OFF -DENABLE_SERIAL=ON
make -j$(nproc)
```

## Build Options

- `ENABLE_CUDA`: enable the CUDA backend.
- `ENABLE_OPENMP`: enable the OpenMP backend.
- `ENABLE_SERIAL`: enable the Kokkos Serial backend.

At least one backend must be enabled.

## Notes

The root `CMakeLists.txt` currently sets CUDA-related compiler variables before
`project()`. If your CUDA host compiler has a different name or path, adjust the
CMake configuration accordingly.

