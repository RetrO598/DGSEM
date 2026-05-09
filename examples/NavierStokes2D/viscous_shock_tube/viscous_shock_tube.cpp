#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <dgsem.hpp>
#include <iostream>

template <class T>
KOKKOS_INLINE_FUNCTION std::array<T, 4>
daru_tenaud_state(const std::array<T, 2>& coordinate, T gamma) {
  if (coordinate[0] <= static_cast<T>(0.5)) {
    return DGSEM::utils::prim_to_cons(
        std::array<T, 4>{static_cast<T>(120.0), T{0}, T{0},
                         static_cast<T>(120.0) / gamma},
        gamma);
  }

  return DGSEM::utils::prim_to_cons(
      std::array<T, 4>{static_cast<T>(1.2), T{0}, T{0},
                       static_cast<T>(1.2) / gamma},
      gamma);
}

template <class T>
struct DaruTenaudInitial
    : public DGSEM::AbstractInitial<
          DaruTenaudInitial<T>,
          DGSEM::equations::CompressibleNavierStokes2D<T>> {
  using Eq = DGSEM::equations::CompressibleNavierStokes2D<T>;
  inline constexpr static std::size_t NDIMS = Eq::NDIMS;
  inline constexpr static std::size_t NVARS = Eq::NVARS;

  explicit DaruTenaudInitial(T gamma_) : gamma(gamma_) {}

  KOKKOS_INLINE_FUNCTION
  std::array<T, NVARS> operator()(std::array<T, NDIMS> coordinate) const {
    return daru_tenaud_state(coordinate, gamma);
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

    std::size_t nx = 800;
    std::size_t ny = 400;
    value_type t_final = 1.0;
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
    constexpr value_type prandtl = 0.73;
    constexpr value_type mu = 1.0e-3;

    DGSEM::BasisGuard<MyBasis> basis;

    auto boundaries = DGSEM::BoundarySet(
        DGSEM::NSNoSlipAdiabaticWallBC{}, DGSEM::NSNoSlipAdiabaticWallBC{},
        DGSEM::NSNoSlipAdiabaticWallBC{}, DGSEM::NSSlipAdiabaticWallBC{});

    const std::array<value_type, 2> domain_left = {0.0, 0.0};
    const std::array<value_type, 2> domain_right = {1.0, 0.5};
    const std::array<std::size_t, 2> n_cells = {nx, ny};

    Eq eq(gamma, mu, prandtl);
    auto problem =
        DGSEM::make_structured_problem<MyBasis, VolumeFlux, SurfaceFlux>(
            eq, domain_left, domain_right, n_cells, boundaries, {false, false});
    auto& solver = problem.solver();
    auto& sol = problem.solution();
    auto& container = problem.elements();
    const auto& mesh = problem.mesh();
    using Solution = typename decltype(problem)::SolutionType;

    solver.set_indicator_parameters(1.0, 0.1, false);

    DaruTenaudInitial<value_type> initial{gamma};
    problem.initialize(initial);

    using Analyzer =
        DGSEM::AnalyzerWrapper<MyBasis, Eq,
                               DGSEM::DivergenceChecker<value_type, Eq::NVARS>>;
    Analyzer analyzer;

    using VTUOutputObserver =
        DGSEM::VTUOutputObserver<value_type, MyBasis, Solution,
                                 decltype(container.node_coordinates), Eq>;

    const value_type cfl = 0.4;
    const value_type dx = (domain_right[0] - domain_left[0]) / nx;
    const value_type dy = (domain_right[1] - domain_left[1]) / ny;
    const value_type max_wave_speed = 7.0;
    const value_type dt =
        cfl * std::min(dx, dy) /
        ((2.0 * static_cast<value_type>(order) + 1.0) * max_wave_speed);

    std::cout << "Daru-Tenaud viscous shock tube: nx=" << nx << ", ny=" << ny
              << ", mu=" << mu << ", dt=" << dt << ", t_final=" << t_final
              << '\n';

    auto time_integrator = problem.make_ssprk3(t_final);
    time_integrator.add_observer(DGSEM::make_analysis_observer<MyBasis, Eq>(
        DGSEM::PointwiseAnalysisTag{}, analyzer, sol, n_cells,
        DGSEM::StopOnNaN<Eq>()));
    time_integrator.add_observer(std::make_unique<DGSEM::PrintObserver>(1000));
    time_integrator.add_observer(std::make_unique<VTUOutputObserver>(
        "navier_stokes_2d_viscous_shock_tube", sol, container.node_coordinates,
        n_cells, 1000));

    time_integrator.solve(solver, sol, dt);
  }
  return 0;
}
