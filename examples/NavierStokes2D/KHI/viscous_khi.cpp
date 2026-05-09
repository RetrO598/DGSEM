#include "boundary_condition/periodic_boundary.hpp"
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <dgsem.hpp>
#include <iostream>
#include <numbers>

template <class T>
KOKKOS_INLINE_FUNCTION std::array<T, 4>
kelvin_hel(const std::array<T, 2>& coordinate, T gamma) {
  T slope = 15.0;
  T B = std::tanh(slope * coordinate[1] + 7.5) -
        std::tanh(slope * coordinate[1] - 7.5);
  T rho = 0.5 + 0.75 * B;
  T v1 = 0.5 * (B - 1);
  T v2 = 0.1 * std::sin(2.0 * std::numbers::pi * coordinate[0]);
  T p = 1.0;

  return DGSEM::utils::prim_to_cons(std::array<T, 4>{rho, v1, v2, p}, gamma);
}

template <class T>
struct KelvinHelmholtzInitial
    : public DGSEM::AbstractInitial<
          KelvinHelmholtzInitial<T>,
          DGSEM::equations::CompressibleNavierStokes2D<T>> {
  using Eq = DGSEM::equations::CompressibleNavierStokes2D<T>;
  inline constexpr static std::size_t NDIMS = Eq::NDIMS;
  inline constexpr static std::size_t NVARS = Eq::NVARS;

  explicit KelvinHelmholtzInitial(T gamma_) : gamma(gamma_) {}

  KOKKOS_INLINE_FUNCTION
  std::array<T, NVARS> operator()(std::array<T, NDIMS> coordinate) const {
    return kelvin_hel(coordinate, gamma);
  }

  T gamma;
};

int main(int argc, char* argv[]) {
  DGSEM::KokkosSession kokkos(argc, argv);
  {
    using value_type = double;
    using Eq = DGSEM::equations::CompressibleNavierStokes2D<value_type>;
    constexpr std::size_t order = 3;
    using MyBasis = DGSEM::Basis::LobattoLegendreBasis<value_type, order>;
    using SurfaceFlux = DGSEM::LaxFriedrichsFlux<Eq>;
    using VolumeFlux = DGSEM::VolumeIntegralShockCapturingHG<
        MyBasis, Eq, DGSEM::ChandrashekarFlux, DGSEM::LaxFriedrichsFlux,
        DGSEM::HGIndicator<MyBasis, Eq>>;

    std::size_t nx = 256;
    std::size_t ny = 256;
    value_type t_final = 10.0;
    if (argc > 1) {
      nx = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    }
    if (argc > 2) {
      ny = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
    }
    if (argc > 3) {
      t_final = std::strtod(argv[3], nullptr);
    }

    constexpr value_type gamma = 1.4;
    constexpr value_type prandtl = 0.72;
    constexpr value_type mu = 1.0e-4;

    DGSEM::BasisGuard<MyBasis> basis;

    auto boundaries =
        DGSEM::BoundarySet(DGSEM::PeriodicBC{}, DGSEM::PeriodicBC{},
                           DGSEM::PeriodicBC{}, DGSEM::PeriodicBC{});

    const std::array<value_type, 2> domain_left = {-1.0, -1.0};
    const std::array<value_type, 2> domain_right = {1.0, 1.0};
    const std::array<std::size_t, 2> n_cells = {nx, ny};

    Eq eq(gamma, mu, prandtl);
    auto problem =
        DGSEM::make_structured_problem<MyBasis, VolumeFlux, SurfaceFlux>(
            eq, domain_left, domain_right, n_cells, boundaries, {true, true});
    auto& solver = problem.solver();
    auto& sol = problem.solution();
    auto& container = problem.elements();
    const auto& mesh = problem.mesh();
    using Solution = typename decltype(problem)::SolutionType;

    solver.set_indicator_parameters(0.5, 0.001, false);

    KelvinHelmholtzInitial<value_type> initial{gamma};
    problem.initialize(initial);

    using Analyzer =
        DGSEM::AnalyzerWrapper<MyBasis, Eq,
                               DGSEM::DivergenceChecker<value_type, Eq::NVARS>>;
    Analyzer analyzer;

    using VTUOutputObserver =
        DGSEM::VTUOutputObserver<value_type, MyBasis, Solution,
                                 decltype(container.node_coordinates), Eq>;

    const value_type cfl = 0.5;
    const value_type dx = (domain_right[0] - domain_left[0]) / nx;
    const value_type dy = (domain_right[1] - domain_left[1]) / ny;
    const value_type max_wave_speed = 2.5;
    const value_type dt =
        cfl * std::min(dx, dy) /
        ((2.0 * static_cast<value_type>(order) + 1.0) * max_wave_speed);

    std::cout << "Viscous Kelvin-Helmholtz: nx=" << nx << ", ny=" << ny
              << ", mu=" << mu << ", dt=" << dt << ", t_final=" << t_final
              << '\n';

    auto time_integrator = problem.make_ssprk3(t_final);
    time_integrator.add_observer(DGSEM::make_analysis_observer<MyBasis, Eq>(
        DGSEM::PointwiseAnalysisTag{}, analyzer, sol, n_cells,
        DGSEM::StopOnNaN<Eq>()));
    time_integrator.add_observer(std::make_unique<DGSEM::PrintObserver>(500));
    time_integrator.add_observer(std::make_unique<VTUOutputObserver>(
        "navier_stokes_2d_khi", sol, container.node_coordinates, n_cells,
        2000));

    time_integrator.solve(solver, sol, dt);
  }
  return 0;
}
