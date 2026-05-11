# Architecture

DGSEM is organized as a mostly header-based C++20 library under `src/`.

## Module Map

- `src/dgsem.hpp`: aggregate include for public components.
- `src/base/`: traits, fixed-size containers, mapping, numerical fluxes.
- `src/mesh/`: mesh types, currently centered on `StructuredMesh`.
- `src/containers/`: solution and element geometry containers.
- `src/equations/`: physical models such as Euler, Navier-Stokes, scalar laws.
- `src/boundary_condition/`: boundary condition implementations and dispatch.
- `src/space_integral/`: volume, surface, gradient, viscous, and Jacobian terms.
- `src/solver/`: `StructuredSolver`, the spatial operator coordinator.
- `src/time_integrator/`: time integration, currently including `SSPRK3`.
- `src/observer/`: output and runtime observation hooks.
- `src/analyzer/`: diagnostics and analysis helpers.

## Ownership Flow

```text
StructuredMesh
  describes domain bounds and cell counts

StructuredElementContainer
  owns nodal coordinates, metrics, Jacobians, neighbors

Solution
  owns state and RHS arrays

StructuredSolver
  references equation, mesh, element data, boundary set
  coordinates volume, interface, boundary, and surface terms

TimeIntegrator
  owns time-stepping work buffers
  calls solver.calc_rhs(...)
```

## Template Composition

Most solver behavior is selected at compile time:

```cpp
StructuredSolver<Eq, Basis, VolumeFlux, SurfaceFlux, Mesh, BoundarySet>
```

This keeps runtime overhead low, but makes examples and documentation important:
users need to know which types are meant to be combined.

