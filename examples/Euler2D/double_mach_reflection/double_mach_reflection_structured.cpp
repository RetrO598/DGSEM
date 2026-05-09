#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <dgsem.hpp>
#include <numbers>

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
  DGSEM::KokkosSession kokkos(argc, argv);
  {
    using value_type = double;
    using Eq = DGSEM::equations::CompressibleEuler2D<value_type>;
    using MyBasis = DGSEM::Basis::LobattoLegendreBasis<value_type, 4>;
    using SurfaceFlux = DGSEM::LaxFriedrichsFlux<Eq>;
    using VolumeFlux = DGSEM::VolumeIntegralShockCapturingHG<
        MyBasis, Eq, DGSEM::ChandrashekarFlux, DGSEM::LaxFriedrichsFlux,
        DGSEM::HGIndicator<MyBasis, Eq>>;

    DGSEM::BasisGuard<MyBasis> basis;

    constexpr value_type gamma = 1.4;
    DoubleMachReflectionBoundaryState<value_type> inflow{gamma};
    BottomInflowRegion<value_type> bottom_inflow_region{};

    auto boundaries = DGSEM::BoundarySet(
        DGSEM::DirichletBC(inflow), DGSEM::OutflowBC{},
        DGSEM::MixedDirichletSlipWallBC(inflow, bottom_inflow_region),
        DGSEM::DirichletBC(inflow));

    std::size_t nx = 480;
    std::size_t ny = 120;
    value_type t_final = 0.2;
    if (argc > 1)
      nx = static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10));
    if (argc > 2)
      ny = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
    if (argc > 3)
      t_final = std::strtod(argv[3], nullptr);

    std::array<value_type, 2> domain_left = {0.0, 0.0};
    std::array<value_type, 2> domain_right = {4.0, 1.0};
    // std::array<value_type, 4> domain_mesh = {0.0, 4.0, 0.0, 1.0};
    // std::array<std::array<value_type, 2>, 2> mapping_domain = {domain_left,
    //                                                            domain_right};
    std::array<std::size_t, 2> n_cells = {nx, ny};

    Eq eq{gamma};
    auto problem =
        DGSEM::make_structured_problem<MyBasis, VolumeFlux, SurfaceFlux>(
            eq, domain_left, domain_right, n_cells, boundaries, {false, false});
    auto& solver = problem.solver();
    auto& sol = problem.solution();
    auto& container = problem.elements();
    const auto& mesh = problem.mesh();
    using Solution = typename decltype(problem)::SolutionType;

    solver.set_indicator_parameters(0.5, 0.001, false);

    DoubleMachReflectionInitial<value_type> initial{gamma};
    problem.initialize(initial);

    using Analyzer =
        DGSEM::AnalyzerWrapper<MyBasis, Eq,
                               DGSEM::DivergenceChecker<value_type, Eq::NVARS>>;
    Analyzer analyzer;

    using DGSEM::PrintObserver;
    using VTUOutputObserver =
        DGSEM::VTUOutputObserver<value_type, MyBasis, Solution,
                                 decltype(container.node_coordinates), Eq>;

    auto time_integrator = problem.make_ssprk3(t_final);

    const value_type cfl = 0.2;
    const value_type dx = (domain_right[0] - domain_left[0]) / nx;
    const value_type dy = (domain_right[1] - domain_left[1]) / ny;
    const value_type max_speed = 12.0;
    const value_type dt =
        cfl * std::min(dx, dy) / ((2.0 * MyBasis::NNodes - 1.0) * max_speed);

    time_integrator.add_observer(DGSEM::make_analysis_observer<MyBasis, Eq>(
        DGSEM::PointwiseAnalysisTag{}, analyzer, sol, n_cells,
        DGSEM::StopOnNaN<Eq>()));

    time_integrator.add_observer(std::make_unique<PrintObserver>(500));
    time_integrator.add_observer(std::make_unique<VTUOutputObserver>(
        "double_mach_reflection_structured_output", sol,
        container.node_coordinates, n_cells));
    time_integrator.solve(solver, sol, dt);
  }
  return 0;
}
