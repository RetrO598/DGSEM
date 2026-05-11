# DGSEM Documentation

This directory contains the project documentation for DGSEM, a high-order
DGSEM (Discontinuous Galerkin Spectral Element Method) CFD solver built with C++20 and Kokkos.
Aiming at facilitating the development and implementation of new numerical schemes, the solver 
currently supports uniform structured meshes only. However, the unstructured curvilinear solver 
will be developed in the future. 

*This framework is still in the initial development phase.*

The documentation is intentionally split into user-facing guides and
developer-facing notes. The goal is to explain how the template components fit
together, not only to list API symbols.

## Contents

- [Getting Started](getting_started.md): run and understand a first example.
- [Build Guide](build.md): configure CPU/OpenMP and CUDA builds.
- [Architecture](architecture.md): major modules and ownership flow.
- [Solver Pipeline](solver_pipeline.md): how `StructuredSolver::calc_rhs()` is
  assembled.
- [Examples Guide](examples.md): overview of available example cases.
- [Extending Equations](extending_equations.md): add a new physical model.
- [Extending Boundary Conditions](extending_boundary_conditions.md): add or
  customize boundary behavior.
- [CUDA and Kokkos Notes](cuda_notes.md): device-code conventions and common
  pitfalls.
- [API Reference Plan](api_reference.md): notes for future Doxygen/Sphinx API
  generation.

## Suggested Reading Order

1. Start with [Build Guide](build.md).
2. Run one example from [Getting Started](getting_started.md).
3. Read [Architecture](architecture.md) and [Solver Pipeline](solver_pipeline.md)
   before modifying solver internals.
4. Use the extension guides when adding equations, fluxes, or boundary
   conditions.

