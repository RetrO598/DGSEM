#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <dgsem.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>

template <class T>
KOKKOS_INLINE_FUNCTION std::array<T, 4>
double_mach_state(const std::array<T, 2>& coordinate, T time, T gamma) {
  constexpr T one_sixth = static_cast<T>(1.0 / 6.0);
  const T shock_position =
      one_sixth + (coordinate[1] + static_cast<T>(20.0) * time) /
                      std::sqrt(static_cast<T>(3.0));

  if (coordinate[0] < shock_position) {
    constexpr T phi = static_cast<T>(std::numbers::pi / 6.0);
    const T sin_phi = std::sin(phi);
    const T cos_phi = std::cos(phi);
    return DGSEM::utils::prim_to_cons(
        std::array<T, 4>{static_cast<T>(8.0), static_cast<T>(8.25) * cos_phi,
                         -static_cast<T>(8.25) * sin_phi,
                         static_cast<T>(116.5)},
        gamma);
  }

  return DGSEM::utils::prim_to_cons(
      std::array<T, 4>{static_cast<T>(1.4), static_cast<T>(0.0),
                       static_cast<T>(0.0), static_cast<T>(1.0)},
      gamma);
}

template <class T>
struct DoubleMachReflectionInitial
    : public DGSEM::AbstractInitial<DoubleMachReflectionInitial<T>,
                                    DGSEM::equations::CompressibleEuler2D<T>> {
  using Eq = DGSEM::equations::CompressibleEuler2D<T>;
  inline constexpr static std::size_t NDIMS = Eq::NDIMS;
  inline constexpr static std::size_t NVARS = Eq::NVARS;

  explicit DoubleMachReflectionInitial(T gamma_) : gamma(gamma_) {}

  KOKKOS_INLINE_FUNCTION
  std::array<T, NVARS> operator()(std::array<T, NDIMS> coordinate) const {
    return double_mach_state(coordinate, static_cast<T>(0.0), gamma);
  }

  T gamma;
};

template <class T>
struct DoubleMachReflectionBoundaryState {
  KOKKOS_INLINE_FUNCTION std::array<T, 4>
  operator()(const std::array<T, 2>& coord, T time) const {
    return double_mach_state(coord, time, gamma);
  }

  T gamma;
};

template <class T>
struct BottomInflowRegion {
  KOKKOS_INLINE_FUNCTION bool operator()(const std::array<T, 2>& coord,
                                         T /*time*/) const {
    return coord[0] < static_cast<T>(1.0 / 6.0);
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    using value_type = double;
    using Eq = DGSEM::equations::CompressibleEuler2D<value_type>;
    using MyBasis = DGSEM::Basis::LobattoLegendreBasis<value_type, 4>;
    using SurfaceFlux = DGSEM::LaxFriedrichsFlux<Eq>;
    using VolumeFlux = DGSEM::VolumeIntegralShockCapturingHG<
        MyBasis, Eq, DGSEM::ChandrashekarFlux, DGSEM::LaxFriedrichsFlux,
        DGSEM::HGIndicator<MyBasis, Eq>>;
    using Mesh = DGSEM::StructuredMesh<value_type, 2>;

    MyBasis::initialize();

    constexpr value_type gamma = 1.4;
    DoubleMachReflectionBoundaryState<value_type> inflow{gamma};
    BottomInflowRegion<value_type> bottom_inflow_region{};

    auto boundaries = DGSEM::BoundarySet(
        DGSEM::DirichletBC(inflow), DGSEM::OutflowBC{},
        DGSEM::MixedDirichletSlipWallBC(inflow, bottom_inflow_region),
        DGSEM::DirichletBC(inflow));
    using Solver = DGSEM::StructuredSolver<Eq, MyBasis, VolumeFlux, SurfaceFlux,
                                           Mesh, decltype(boundaries)>;
    using Solution = DGSEM::Solution<Mesh, MyBasis, Eq>;

    std::size_t nx = 480;
    std::size_t ny = 120;
    value_type t_final = 0.2;
    std::string output_path = "double_mach_reflection_structured.txt";
    if (argc > 1)
      nx = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    if (argc > 2)
      ny = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
    if (argc > 3)
      t_final = std::strtod(argv[3], nullptr);
    if (argc > 4)
      output_path = argv[4];

    std::array<value_type, 4> domain_mesh = {0.0, 4.0, 0.0, 1.0};
    std::array<std::array<value_type, 2>, 2> mapping_domain = {
        std::array<value_type, 2>{0.0, 0.0},
        std::array<value_type, 2>{4.0, 1.0}};
    std::array<std::size_t, 2> n_cells = {nx, ny};

    Mesh mesh(domain_mesh, n_cells);
    Eq eq{gamma};

    DGSEM::StructuredElementContainer<value_type, 2> container;
    DGSEM::StructuredElementInitializer<
        value_type, MyBasis, DGSEM::LinearMapping<std::array<value_type, 2>>, 2>
        initializer{DGSEM::LinearMapping<std::array<value_type, 2>>(
                        mapping_domain[0], mapping_domain[1]),
                    {false, false}};

    initializer.init_elements(n_cells, container);

    Solver solver(eq, mesh, container, boundaries);
    solver.set_indicator_parameters(0.5, 0.001, false);

    Solution sol(mesh);
    DoubleMachReflectionInitial<value_type> initial{gamma};
    solver.initialize(initial, sol);

    using TimeIntegrator = DGSEM::SSPRK3<value_type, Solver, Mesh, Solution>;
    TimeIntegrator time_integrator(sol, mesh);

    const value_type cfl = 0.2;
    const value_type dx = (domain_mesh[1] - domain_mesh[0]) / nx;
    const value_type dy = (domain_mesh[3] - domain_mesh[2]) / ny;
    const value_type max_speed = 12.0;

    int iter = 0;
    value_type t = 0.0;
    while (t < t_final) {
      const value_type dt =
          cfl * std::min(dx, dy) / ((2.0 * MyBasis::NNodes - 1.0) * max_speed);
      const value_type dt_step = std::min(dt, t_final - t);
      time_integrator.step(solver, sol, dt_step);
      t += dt_step;
      ++iter;

      if (iter % 50 == 0) {
        std::cout << "Iter: " << std::setw(6) << iter
                  << "  Time: " << std::fixed << std::setprecision(6) << t
                  << std::endl;
      }
    }

    auto u_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sol.u_device);
    auto coord_host = container.node_coordinates;

    std::ofstream solution_file(output_path, std::ios::out);
    solution_file << std::scientific << std::setprecision(16);
    const std::size_t ndofs = MyBasis::NNodes * MyBasis::NNodes;

    for (std::size_t ielem = 0; ielem < n_cells[0]; ++ielem) {
      for (std::size_t jelem = 0; jelem < n_cells[1]; ++jelem) {
        for (std::size_t dof = 0; dof < ndofs; ++dof) {
          const std::array<value_type, 4> u_cons = {
              u_host(ielem, jelem, dof, 0), u_host(ielem, jelem, dof, 1),
              u_host(ielem, jelem, dof, 2), u_host(ielem, jelem, dof, 3)};
          const auto u_prim = DGSEM::utils::cons_to_prim(u_cons, eq);
          solution_file << coord_host(ielem, jelem, dof, 0) << " "
                        << coord_host(ielem, jelem, dof, 1) << " " << u_prim[0]
                        << " " << u_prim[1] << " " << u_prim[2] << " "
                        << u_prim[3] << "\n";
        }
      }
    }

    MyBasis::finalize();
  }
  Kokkos::finalize();
  return 0;
}
