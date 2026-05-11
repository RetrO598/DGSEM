# Getting Started

The fastest way to understand DGSEM is to inspect and run the 2D Euler blast wave example:

```text
examples/Euler2D/blast_wave_hg/blast_wave_hg.cpp
```

## Minimal Simulation Flow

Most examples follow this pattern:

```cpp
DGSEM::KokkosSession kokkos;
{
  using value_type = double;

  // Declare the equations you want to solve.
  using Eq = DGSEM::equations::CompressibleEuler2D<value_type>;

  // Define the polynomial order using a template parameter, here Order = 3,
  // which means we define four solution points in each coordinate direction.
  using Basis = DGSEM::Basis::LobattoLegendreBasis<value_type, 3>;

  // Define the numerical flux for surface integral.
  using SurfaceFlux = DGSEM::LaxFriedrichsFlux<Eq>;

  // Define the volume integral policy. This example uses the HG
  // shock-capturing volume integral with Chandrashekar's two-point flux and
  // Lax-Friedrichs surface flux.
  using VolumeFlux = DGSEM::VolumeIntegralShockCapturingHG<
      Basis, Eq, DGSEM::ChandrashekarFlux, DGSEM::LaxFriedrichsFlux,
      DGSEM::HGIndicator<Basis, Eq>>;

  using Mesh = DGSEM::StructuredMesh<value_type, 2>;

  // KokkosSession has already initialized Kokkos. BasisGuard initializes the
  // static basis views and finalizes them automatically before Kokkos shuts
  // down.
  DGSEM::BasisGuard<Basis> basis;

  // BoundarySet stores one boundary condition for each structured-mesh face.
  // In 2D the face order is left, right, bottom, top. This periodic blast-wave
  // case therefore provides four periodic boundary conditions.
  auto boundaries =
      DGSEM::BoundarySet(DGSEM::PeriodicBC{}, DGSEM::PeriodicBC{},
                         DGSEM::PeriodicBC{}, DGSEM::PeriodicBC{});

  // The solver type is fully determined at compile time by the physical
  // equation, polynomial basis, volume integral, surface flux, mesh, and
  // boundary set. This avoids runtime dispatch in the hot path.
  using Solver = DGSEM::StructuredSolver<Eq, Basis, VolumeFlux, SurfaceFlux,
                                         Mesh, decltype(boundaries)>;

  // Solution owns the conservative variables, RHS storage, and surface flux
  // storage for the selected mesh/basis/equation combination.
  using Solution = DGSEM::Solution<Mesh, Basis, Eq>;

  // The blast-wave example uses a square domain and the same number of cells in
  // both directions. Smaller values are useful for quick smoke tests.
  std::size_t nx = 256;
  std::size_t ny = 256;
  value_type t_final = 1.2;

  std::array<value_type, 2> domain_left = {-2.0, -2.0};
  std::array<value_type, 2> domain_right = {2.0, 2.0};
  std::array<std::size_t, 2> n_cells = {nx, ny};

  // Define the structured mesh used to solve the problem. The dimension of the
  // mesh is prescribed through a template parameter. 'domain_left' and
  // 'domain_right' should be std::array to provide the physical domain of
  // computation. 'n_cells' is also a std::array to provide the number of cells
  // in each direction.
  Mesh mesh(domain_left, domain_right, n_cells);

  // The Euler equation object stores physical parameters, here the ideal-gas
  // ratio of specific heats.
  Eq eq{1.4};

  // This special container stores the important information of meshes, for
  // examples node coordinates, jacobian matrix and contravariant vectors of
  // element transfer.
  DGSEM::StructuredElementContainer<value_type, 2> elements;

  // This is used to compute and initialize the data in the above container.
  DGSEM::StructuredElementInitializer<
      value_type, Basis, DGSEM::LinearMapping<std::array<value_type, 2>>, 2>
      initializer{DGSEM::LinearMapping<std::array<value_type, 2>>(
                      domain_left, domain_right),
                  {true, true}};
  initializer.init_elements(n_cells, elements);

  // The solver keeps the equation, mesh, element metrics, and boundary
  // conditions together and provides calc_rhs(...) for the time integrator.
  Solver solver(eq, mesh, elements, boundaries);

  // Configure the HG shock-capturing indicator. The first two parameters bound
  // the artificial-viscosity strength, and the last flag controls smoothing.
  solver.set_indicator_parameters(0.5, 0.001, false);

  // Allocate solution arrays on the active Kokkos execution space.
  Solution sol(mesh);

  // Initial conditions are written as device-callable functors. The actual
  // BlastWaveInitial definition is in the full example source file.
  BlastWaveInitial<value_type> initial_condition{};
  solver.initialize(initial_condition, sol);

  // SSPRK3 owns two temporary solution buffers with the same shape as sol.
  // Construct it once outside the time loop or solve(...) call.
  DGSEM::SSPRK3<value_type, Solver, Mesh, Solution> time_integrator(
      sol, mesh, t_final);

  // Choose a stable time step from a CFL estimate. The polynomial degree enters
  // through Basis::NNodes, so higher-order runs use a smaller dt.
  const value_type cfl = 0.5;
  const value_type dx = (domain_right[0] - domain_left[0]) / nx;
  const value_type dy = (domain_right[1] - domain_left[1]) / ny;
  const value_type max_speed = 2.0;
  const value_type dt =
      cfl * std::min(dx, dy) / ((2.0 * Basis::NNodes - 1.0) * max_speed);

  // Advance until t_final. Observers can be attached to time_integrator before
  // this call for printing, analysis, or VTU output.
  time_integrator.solve(solver, sol, dt);
}
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

Basis data must be initialized after Kokkos starts and finalized before Kokkos
shuts down. Prefer `DGSEM::KokkosSession` and `DGSEM::BasisGuard<Basis>` in
application code so the teardown order is automatic.
