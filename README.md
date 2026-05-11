# DGSEM

DGSEM is a C++20 high-order computational fluid dynamics solver based on the
Discontinuous Galerkin Spectral Element Method. The code is organized as a
modular, mostly header-based library and uses Kokkos for performance portability
across CPU and GPU backends.

The current implementation focuses on structured meshes, high-order nodal
polynomial bases, compressible Euler and Navier-Stokes equations, numerical
fluxes, shock-capturing volume integrals, and explicit Runge-Kutta time
integration.

## Documentation

The project documentation is published with GitHub Pages:

[https://retro598.github.io/DGSEM/](https://retro598.github.io/DGSEM/)

## Algorithm Overview

DGSEM approximates conservation laws by representing the solution inside each
element with high-order nodal polynomials. Volume terms are evaluated at
collocation nodes, while neighboring elements communicate through numerical
surface fluxes. This gives a compact, high-order method with element-local data
structures that are well suited to parallel execution.

This project follows the split-form DGSEM philosophy for compressible flow
problems. Split forms and summation-by-parts structure help reduce aliasing
errors and improve nonlinear robustness for under-resolved simulations. For
flows containing shocks or strong gradients, the code also includes
shock-capturing method based on hybrid high-order/low-order subcell ideas,
with troubled-cell indicators controlling the amount of added dissipation.

## Features

- C++20 template-based solver components.
- Kokkos-based CPU/GPU portability.
- Structured mesh support.
- Lobatto-Legendre nodal basis.
- Compressible Euler models in 1D, 2D, and 3D.
- Compressible Navier-Stokes models in 2D and 3D.
- Surface fluxes such as Lax-Friedrichs and HLLC.
- Split-form and shock-capturing volume integral options.
- SSPRK3 explicit time integration.
- VTU output and analysis observers.

## Quick Build

CPU/OpenMP:

```bash
mkdir build && cd build
cmake .. -DENABLE_CUDA=OFF -DENABLE_OPENMP=ON -DENABLE_SERIAL=ON
make -j$(nproc)
```

CUDA:

```bash
mkdir build-cuda && cd build-cuda
cmake .. -DENABLE_CUDA=ON -DENABLE_OPENMP=OFF -DENABLE_SERIAL=ON
make -j$(nproc)
```

See the [Build Guide](doc/build.md) for more details.

## Examples

Example cases are located in [`examples/`](examples/), including:

- 1D Euler shock tube and Shu-Osher problems.
- 2D Euler blast wave, Kelvin-Helmholtz, Riemann, double Mach reflection, and
  astro jet problems.
- 3D Euler Taylor-Green vortex and Sedov blast wave cases.
- 2D/3D Navier-Stokes examples.
- 1D scalar conservation-law examples.

## References

The numerical approach in this project is closely related to the following
works:

1. Sebastian Hennemann, Andrés M. Rueda-Ramírez, Florian J. Hindenlang, and
   Gregor J. Gassner. "A provably entropy stable subcell shock capturing
   approach for high order split form DG for the compressible Euler equations."
   *Journal of Computational Physics*, 426, 109935, 2021.
   <https://doi.org/10.1016/j.jcp.2020.109935>

2. Gregor J. Gassner, Andrew R. Winters, and David A. Kopriva. "Split form nodal
   discontinuous Galerkin schemes with summation-by-parts property for the
   compressible Euler equations." *Journal of Computational Physics*, 327,
   39-66, 2016. <https://doi.org/10.1016/j.jcp.2016.09.013>

<!-- ## License

Add license information here before publishing a formal release. -->

