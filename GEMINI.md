# DGSEM CFD Solver

## Project Overview
This project is a high-order Discontinuous Galerkin Spectral Element Method (DGSEM) solver for Computational Fluid Dynamics (CFD). It is implemented in C++20 and leverages the **Kokkos** ecosystem for performance portability across different hardware architectures (CPUs, GPUs).

The solver is designed to be modular and flexible, utilizing template metaprogramming to swap between different physical equations, polynomial bases, and numerical flux schemes.

### Key Technologies
- **C++20**: Utilizing modern features like concepts and templates.
- **Kokkos**: Hardware-agnostic parallelism and data management.
- **OpenMP**: Often used as a backend for Kokkos on multi-core CPUs.
- **CMake**: Build system for configuration and dependency management.

## Architecture
The codebase is organized as a modular header-only library located in the `src/` directory.

- `src/dgsem.hpp`: Main entry point that includes all library components.
- `src/base/`: Core definitions, traits, and mathematical utilities.
- `src/mesh/`: Mesh definitions, primarily focused on `StructuredMesh`.
- `src/polynomial_basis/`: High-order polynomial bases, such as `LobattoLegendreBasis`.
- `src/equations/`: Physical models (e.g., `LinearScalarAdvection1D`, `CompressibleEuler1D`).
- `src/solver/`: The core `StructuredSolver` that coordinates the DGSEM algorithm.
- `src/space_integral/`: Implementation of volume and surface integrals, including flux computations.
- `src/time_integrator/`: Temporal discretization schemes like `SSPRK3` (Strong Stability Preserving Runge-Kutta).
- `src/subcell_limiting/`: Shock-capturing techniques and indicators for handling discontinuities.
- `src/containers/`: Specialized data structures for solution states and element data.

## Building and Running

### Prerequisites
- A C++20 compliant compiler.
- CMake (3.15+).
- Kokkos (must be installed and discoverable by `find_package`).
- OpenMP.

### Build Instructions
```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Running Examples
After building, the examples can be found in the `build/examples/` directory. For example:
```bash
./build/examples/Scalar1D/shock_capture/shock_capture
```
This typically generates output files like `solution.txt` and `nodes.txt` which can be used for visualization.

## Development Conventions
- **Template-Heavy**: The solver uses templates extensively to ensure zero-overhead abstraction for different dimensions and equations.
- **Kokkos Integration**: Data is stored in `Kokkos::View` for performance. Ensure all kernel operations are compatible with Kokkos' execution spaces.
- **Modular Fluxes**: Numerical fluxes (e.g., Lax-Friedrichs, Central) are implemented as functors that can be injected into the solver.
- **Header-Only**: Most of the core logic resides in `.hpp` files in the `src/` directory.
- **Concepts**: The codebase uses C++20 concepts (e.g., `std::derived_from`) to enforce interface requirements at compile time.
