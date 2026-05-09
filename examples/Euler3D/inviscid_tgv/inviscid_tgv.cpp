#include "analyzer/volume_average_euler.hpp"
#include "observer/analysis_observer.hpp"
#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <dgsem.hpp>
#include <numbers>

template <class T>
struct InviscidTaylorGreenVortex
    : public DGSEM::AbstractInitial<InviscidTaylorGreenVortex<T>,
                                    DGSEM::equations::CompressibleEuler3D<T>> {
  using Eq = DGSEM::equations::CompressibleEuler3D<T>;
  inline constexpr static std::size_t NDIMS = Eq::NDIMS;
  inline constexpr static std::size_t NVARS = Eq::NVARS;

  KOKKOS_INLINE_FUNCTION
  std::array<T, NVARS> operator()(std::array<T, NDIMS> coordinate) const {
    T x = coordinate[0];
    T y = coordinate[1];
    T z = coordinate[2];

    T rho = 1.0;
    T v1 = std::sin(x) * std::cos(y) * std::cos(z);
    T v2 = -std::cos(x) * std::sin(y) * std::cos(z);
    T v3 = 0.0;
    T p = 100.0 / 1.4 +
          1.0 / 16.0 *
              (std::cos(2.0 * x) * std::cos(2.0 * z) + 2.0 * std::cos(2.0 * y) +
               2.0 * std::cos(2.0 * x) + std::cos(2.0 * y) * std::cos(2.0 * z));

    return DGSEM::utils::prim_to_cons(std::array<T, 5>{rho, v1, v2, v3, p},
                                      static_cast<T>(1.4));
  }
};

int main() {
  DGSEM::KokkosSession kokkos;
  {
    using value_type = double;
    using Eq = DGSEM::equations::CompressibleEuler3D<value_type>;
    const std::size_t order = 4;
    using MyBasis = DGSEM::Basis::LobattoLegendreBasis<value_type, order>;
    using SurfaceFlux = DGSEM::LaxFriedrichsFlux<Eq>;
    using VolumeFlux = DGSEM::VolumeIntegralShockCapturingHG<
        MyBasis, Eq, DGSEM::ChandrashekarFlux, DGSEM::LaxFriedrichsFlux,
        DGSEM::HGIndicator<MyBasis, Eq>>;
    // using VolumeFlux =
    //     DGSEM::VolumeIntegralSplitForm<MyBasis, Eq,
    //     DGSEM::ChandrashekarFlux>;

    DGSEM::BasisGuard<MyBasis> basis;

    auto boundaries = DGSEM::BoundarySet(
        DGSEM::PeriodicBC{}, DGSEM::PeriodicBC{}, DGSEM::PeriodicBC{},
        DGSEM::PeriodicBC{}, DGSEM::PeriodicBC{}, DGSEM::PeriodicBC{});

    std::size_t nx = 32;
    std::size_t ny = 32;
    std::size_t nz = 32;
    value_type t_final = 0.001;

    std::array<value_type, 3> domain_left = {0.0, 0.0, 0.0};
    std::array<value_type, 3> domain_right = {
        2.0 * std::numbers::pi, 2.0 * std::numbers::pi, 2.0 * std::numbers::pi};

    std::array<std::size_t, 3> n_cells = {nx, ny, nz};

    Eq eq{1.4};
    auto problem =
        DGSEM::make_structured_problem<MyBasis, VolumeFlux, SurfaceFlux>(
            eq, domain_left, domain_right, n_cells, boundaries,
            {true, true, true});
    auto& solver = problem.solver();
    auto& sol = problem.solution();
    auto& container = problem.elements();
    const auto& mesh = problem.mesh();
    using Solution = typename decltype(problem)::SolutionType;

    solver.set_indicator_parameters(1.0, 0.001, false);

    InviscidTaylorGreenVortex<value_type> initial{};

    problem.initialize(initial);

    using Analyzer =
        DGSEM::AnalyzerWrapper<MyBasis, Eq,
                               DGSEM::DivergenceChecker<value_type, Eq::NVARS>>;
    Analyzer analyzer;

    using static_analyzer =
        DGSEM::AnalyzerWrapper<MyBasis, Eq, DGSEM::VolumeAverageEuler<Eq>>;

    static_analyzer analyzer_statics;
    using DGSEM::PrintObserver;
    using VTUOutputObserver =
        DGSEM::VTUOutputObserver<value_type, MyBasis, Solution,
                                 decltype(container.node_coordinates), Eq>;

    auto time_integrator = problem.make_ssprk3(t_final);

    const value_type cfl = 0.1;
    const value_type dx = (domain_right[0] - domain_left[0]) / nx;
    const value_type dy = (domain_right[1] - domain_left[1]) / ny;
    const value_type dz = (domain_right[2] - domain_left[2]) / nz;
    const value_type max_speed = 2.0;
    const value_type dt =
        cfl * std::min({dx, dy, dz}) / max_speed / (3.0 * (2 * order + 1));
    time_integrator.add_observer(DGSEM::make_analysis_observer<MyBasis, Eq>(
        DGSEM::PointwiseAnalysisTag{}, analyzer, sol, n_cells,
        DGSEM::StopOnNaN<Eq>()));

    time_integrator.add_observer(std::make_unique<PrintObserver>(100));

    time_integrator.add_observer(std::make_unique<VTUOutputObserver>(
        "inviscid_tgv", sol, container.node_coordinates, n_cells));

    time_integrator.add_observer(DGSEM::make_analysis_observer<MyBasis, Eq>(
        DGSEM::VolumeWeightedAnalysisTag{}, analyzer_statics, sol, container,
        n_cells, DGSEM::VolumeAverageCsvWriter<Eq>("static.csv")));

    time_integrator.solve(solver, sol, dt);
  }
  return 0;
}
