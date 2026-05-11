# Examples Guide

Examples live under `examples/` and are currently the main executable tests.

## Example Groups

- `Euler1D/`: 1D Euler shock tube and Shu-Osher cases.
- `Euler2D/`: blast wave, Kelvin-Helmholtz, Riemann, double Mach reflection,
  astro jet.
- `Euler3D/`: inviscid Taylor-Green vortex and Sedov blast wave.
- `NavierStokes2D/`: viscous shock tube and Kelvin-Helmholtz instability.
- `NavierStokes3D/`: Taylor-Green vortex.
- `Scalar1D/`: scalar advection, Burgers, Buckley-Leverett, shock capturing.

## What To Learn From Examples

- How equations, bases, fluxes, and boundaries are combined.
- How element geometry is initialized.
- How observers and VTU output are attached.
- How CFL-based time steps are chosen.

## Testing Role

The examples are useful smoke tests, but they are not a replacement for smaller
unit or regression tests. Future test work should add focused tests for basis
initialization, geometry metrics, boundary dispatch, and short-step solutions.

