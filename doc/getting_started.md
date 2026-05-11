# Getting Started

The fastest way to understand DGSEM is to inspect and run the 1D Euler shock
tube example:

```text
examples/Euler1D/shock_tube/shock_tube.cpp
```

## Minimal Simulation Flow

Most examples follow this pattern:

```cpp
Kokkos::initialize();
{
  using Eq = DGSEM::equations::CompressibleEuler1D<double>;
  using Basis = DGSEM::Basis::LobattoLegendreBasis<double, 4>;
  using SurfaceFlux = DGSEM::HLLCFlux<Eq>;
  using VolumeFlux = /* volume integral policy */;

  Basis::initialize();

  DGSEM::StructuredMesh<double, 1> mesh(domain_left, domain_right, n_cells);
  DGSEM::StructuredElementContainer<double, 1> elements;

  DGSEM::StructuredElementInitializer<double, Basis,
                                      DGSEM::LinearMapping<double>, 1>
      initializer(mapping, periodic);
  initializer.init_elements(n_cells, elements);

  DGSEM::StructuredSolver<Eq, Basis, VolumeFlux, SurfaceFlux,
                          decltype(mesh), decltype(boundaries)>
      solver(eq, mesh, elements, boundaries);

  DGSEM::Solution<decltype(mesh), Basis, Eq> sol(mesh);
  solver.initialize(initial_condition, sol);

  DGSEM::SSPRK3<double, decltype(solver), decltype(mesh), decltype(sol)>
      integrator(sol, mesh);

  integrator.step(solver, sol, dt);

  Basis::finalize();
}
Kokkos::finalize();
```

## Key Ideas

- `Equation` defines the physical model.
- `Basis` owns nodal basis data and must be initialized after Kokkos starts.
- `StructuredMesh` stores the logical grid size and domain bounds.
- `StructuredElementContainer` stores geometry, Jacobian, neighbor, and metric
  data.
- `StructuredSolver` assembles the spatial DGSEM operator.
- `Solution` owns state, RHS, surface flux, and optional parabolic work arrays.
- `SSPRK3` advances the solution in time.

## Common Pitfall

Always initialize basis data after `Kokkos::initialize()` and finalize it before
`Kokkos::finalize()`. The basis stores Kokkos views.

