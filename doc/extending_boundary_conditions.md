# Extending Boundary Conditions

Boundary conditions are collected in `BoundarySet`, one entry per mesh face.

## Face Count

Structured grids expect:

- 1D: 2 faces.
- 2D: 4 faces.
- 3D: 6 faces.

`StructuredSolver` checks that `BoundarySet::NFACES == 2 * NDIMS`.

## Implementation Pattern

Boundary condition classes usually provide device-callable `apply_device`
methods. Optional parabolic boundary support can provide gradient and viscous
handlers.

Look at these files first:

- `src/boundary_condition/dirichlet_boundary.hpp`
- `src/boundary_condition/periodic_boundary.hpp`
- `src/boundary_condition/slip_wall_boundary.hpp`
- `src/boundary_condition/navier_stokes_wall_boundary.hpp`
- `src/boundary_condition/boundary_set.hpp`

## Device Compatibility

Boundary logic runs inside Kokkos kernels. Keep methods small, inline, and
device-compatible.

