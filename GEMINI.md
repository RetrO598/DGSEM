# DGSEM CFD Solver

## Project Overview
This project is a high-order Discontinuous Galerkin Spectral Element Method (DGSEM) solver for Computational Fluid Dynamics (CFD). It is implemented in C++20 and leverages the **Kokkos** ecosystem for performance portability across different hardware architectures (CPUs, GPUs).

The solver is designed to be modular and flexible, utilizing template metaprogramming to swap between different physical equations, polynomial bases, and numerical flux schemes.

### Key Technologies
- **C++20**: Utilizing modern features like concepts (`std::derived_from`), `std::span`, and templates.
- **Kokkos**: Hardware-agnostic parallelism and data management via `Kokkos::View`.
- **GPU Support**: Fully compatible with NVIDIA CUDA backends (requires CUDA 12.1+ for C++20 compatibility).
- **CMake**: Build system for configuration and dependency management.

## Architecture
The codebase is organized as a modular header-only library located in the `src/` directory.

- `src/dgsem.hpp`: Main entry point that includes all library components.
- `src/base/`: Core definitions, traits, and mathematical utilities.
- `src/mesh/`: Mesh definitions, primarily focused on `StructuredMesh`.
- `src/polynomial_basis/`: High-order polynomial bases, such as `LobattoLegendreBasis`. Supports GPU-resident basis data.
- `src/equations/`: Physical models (e.g., `LinearScalarAdvection1D`, `CompressibleEuler1D`).
- `src/solver/`: The core `StructuredSolver` that coordinates the DGSEM algorithm.
- `src/space_integral/`: Implementation of volume and surface integrals, including flux computations.
- `src/time_integrator/`: Temporal discretization schemes like `SSPRK3`.
- `src/subcell_limiting/`: Shock-capturing techniques and indicators (e.g., HG Indicator).
- `src/containers/`: Specialized data structures for solution states and element data.

## Building and Running

### Prerequisites
- **C++20 Compiler**: GCC 11+ or Clang 15+.
- **CUDA (Optional for GPU)**: **CUDA 12.1 or higher** is strictly required to support C++20 features (Concepts, etc.) within NVCC.
- **CMake**: 3.28+.
- **Kokkos**: Fetched automatically by CMake from Kokkos 4.7.02. No separate
  `find_package(Kokkos)` installation is required for the default build.

### Build Instructions (CPU/OpenMP)
```bash
mkdir build && cd build
cmake .. -DENABLE_CUDA=OFF -DENABLE_OPENMP=ON -DENABLE_SERIAL=ON
make -j$(nproc)
```

### Build Instructions (GPU/CUDA)
```bash
mkdir build && cd build
cmake .. -DENABLE_CUDA=ON -DENABLE_OPENMP=OFF -DENABLE_SERIAL=ON
make -j$(nproc)
```

## Development Conventions

### 1. Polynomial Basis Lifecycle
Since polynomial basis data (nodes, weights, matrices) resides in `Kokkos::View` for GPU compatibility, users **must** manage their lifecycle:
- **Initialize**: Call `BasisType::initialize()` after `Kokkos::initialize()`.
- **Finalize**: Call `BasisType::finalize()` before `Kokkos::finalize()` to release static View references.

```cpp
Kokkos::initialize();
{
    using MyBasis = DGSEM::Basis::LobattoLegendreBasis<double, 3>;
    MyBasis::initialize(); // Synchronizes basis data to GPU
    // ... simulation ...
    MyBasis::finalize();   // Releases GPU memory
}
Kokkos::finalize();
```

For application code and examples, prefer the RAII helpers when possible:

```cpp
DGSEM::KokkosSession kokkos;
DGSEM::BasisGuard<MyBasis> basis;
```

### 2. GPU Performance Optimizations
- **Coalesced Access**: Basis matrices (like `derivative_split_transpose`) are pre-transposed during initialization to ensure coalesced memory access within CUDA kernels.
- **Inlining**: All computational kernels must be marked with `KOKKOS_INLINE_FUNCTION` to be callable from both Host and Device.
- **Concepts**: The codebase uses C++20 concepts (e.g., `std::derived_from`) to enforce interface requirements. For CUDA builds, ensure NVCC is configured with `--expt-relaxed-constexpr` and `--expt-extended-lambda`.

### 3. Header-Only Structure
Most of the core logic resides in `.hpp` files. Adding new equations or flux schemes involves creating a new header and ensuring it fulfills the required concepts/traits.
