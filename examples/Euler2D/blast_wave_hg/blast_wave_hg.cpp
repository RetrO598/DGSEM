#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <dgsem.hpp>

template <class T>
struct BlastWaveInitial
    : public DGSEM::AbstractInitial<BlastWaveInitial<T>,
                                    DGSEM::equations::CompressibleEuler2D<T>> {
  using Eq = DGSEM::equations::CompressibleEuler2D<T>;
  inline constexpr static std::size_t NDIMS = Eq::NDIMS;
  inline constexpr static std::size_t NVARS = Eq::NVARS;

  KOKKOS_INLINE_FUNCTION
  std::array<T, NVARS> operator()(std::array<T, NDIMS> coordinate) const {
    const T x = coordinate[0];
    const T y = coordinate[1];
    const T r = std::sqrt(x * x + y * y);

    T rho;
    T v1;
    T v2;
    T p;
    if (r > static_cast<T>(0.5)) {
      rho = static_cast<T>(1.0);
      v1 = static_cast<T>(0.0);
      v2 = static_cast<T>(0.0);
      p = static_cast<T>(1.0e-3);
    } else {
      const T phi = std::atan2(y, x);
      rho = static_cast<T>(1.1691);
      v1 = static_cast<T>(0.1882) * std::cos(phi);
      v2 = static_cast<T>(0.1882) * std::sin(phi);
      p = static_cast<T>(1.245);
    }

    return DGSEM::utils::prim_to_cons(std::array<T, 4>{rho, v1, v2, p},
                                      static_cast<T>(1.4));
  }
};

int main() {
  Kokkos::initialize();
  {
    using value_type = double;
    using Eq = DGSEM::equations::CompressibleEuler2D<value_type>;
    using MyBasis = DGSEM::Basis::LobattoLegendreBasis<value_type, 3>;
    using SurfaceFlux = DGSEM::LaxFriedrichsFlux<Eq>;
    using VolumeFlux = DGSEM::VolumeIntegralShockCapturingHG<
        MyBasis, Eq, DGSEM::ChandrashekarFlux, DGSEM::LaxFriedrichsFlux,
        DGSEM::HGIndicator<MyBasis, Eq>>;
    using Mesh = DGSEM::StructuredMesh<value_type, 2>;

    MyBasis::initialize();

    auto boundaries =
        DGSEM::BoundarySet(DGSEM::PeriodicBC{}, DGSEM::PeriodicBC{},
                           DGSEM::PeriodicBC{}, DGSEM::PeriodicBC{});
    using Solver = DGSEM::StructuredSolver<Eq, MyBasis, VolumeFlux, SurfaceFlux,
                                           Mesh, decltype(boundaries)>;
    using Solution = DGSEM::Solution<Mesh, MyBasis, Eq>;

    std::size_t nx = 256;
    std::size_t ny = 256;
    value_type t_final = 1.2;

    std::array<value_type, 2> domain_left = {-2.0, -2.0};
    std::array<value_type, 2> domain_right = {2.0, 2.0};

    // std::array<value_type, 4> domain_mesh = {-2.0, 2.0, -2.0, 2.0};
    // std::array<std::array<value_type, 2>, 2> mapping_domain = {domain_left,
    //                                                            domain_right};
    std::array<std::size_t, 2> n_cells = {nx, ny};

    Mesh mesh(domain_left, domain_right, n_cells);
    Eq eq{1.4};

    DGSEM::StructuredElementContainer<value_type, 2> container;
    DGSEM::StructuredElementInitializer<
        value_type, MyBasis, DGSEM::LinearMapping<std::array<value_type, 2>>, 2>
        initializer{DGSEM::LinearMapping<std::array<value_type, 2>>(
                        domain_left, domain_right),
                    {true, true}};

    initializer.init_elements(n_cells, container);
    // container.sync_to_device();

    Solver solver(eq, mesh, container, boundaries);
    solver.set_indicator_parameters(0.5, 0.001, false);

    Solution sol(mesh);
    BlastWaveInitial<value_type> initial{};
    solver.initialize(initial, sol);

    using TimeIntegrator = DGSEM::SSPRK3<value_type, Solver, Mesh, Solution>;
    using Analyzer = DGSEM::CompositeAnalyzer<
        MyBasis, Eq, DGSEM::DivergenceChecker<value_type, Eq::NVARS>>;
    Analyzer analyzer;
    using AnalyzerObserver =
        DGSEM::AnalyzerObserver<MyBasis, Eq, Solution, Analyzer>;
    using DGSEM::PrintObserver;
    using VTUOutputObserver =
        DGSEM::VTUOutputObserver<value_type, MyBasis, Solution,
                                 decltype(container.node_coordinates), Eq>;

    TimeIntegrator time_integrator(sol, mesh, t_final);

    const value_type cfl = 0.5;
    const value_type dx = (domain_right[0] - domain_left[0]) / nx;
    const value_type dy = (domain_right[1] - domain_left[1]) / ny;
    const value_type max_speed = 2.0;
    const value_type dt =
        cfl * std::min(dx, dy) / ((2.0 * MyBasis::NNodes - 1.0) * max_speed);
    time_integrator.add_observer(
        std::make_unique<AnalyzerObserver>(analyzer, sol, n_cells));
    time_integrator.add_observer(std::make_unique<PrintObserver>(100));
    time_integrator.add_observer(std::make_unique<VTUOutputObserver>(
        "blast_wave_hg_output", sol, container.node_coordinates, n_cells));
    time_integrator.solve(solver, sol, dt);

    MyBasis::finalize();
  }
  Kokkos::finalize();
  return 0;
}
