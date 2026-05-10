#include <Kokkos_Core.hpp>
#include <array>
#include <cstddef>
#include <dgsem.hpp>

template <class T>
struct Riemann2DInitial
    : public DGSEM::AbstractInitial<Riemann2DInitial<T>,
                                    DGSEM::equations::CompressibleEuler2D<T>> {
  using Eq = DGSEM::equations::CompressibleEuler2D<T>;
  inline constexpr static std::size_t NDIMS = Eq::NDIMS;
  inline constexpr static std::size_t NVARS = Eq::NVARS;

  KOKKOS_INLINE_FUNCTION
  std::array<T, NVARS> operator()(std::array<T, NDIMS> coordinate) const {
    const T x = coordinate[0];
    const T y = coordinate[1];

    T rho;
    T v1;
    T v2;
    T p;
    // if (x >= 0.5 && y >= 0.5) {
    //   rho = 0.5313;
    //   v1 = 0.0;
    //   v2 = 0.0;
    //   p = 0.4;
    // } else if (x < 0.5 && y >= 0.5) {
    //   rho = 1.0;
    //   v1 = 0.7276;
    //   v2 = 0.0;
    //   p = 1.0;
    // } else if (x < 0.5 && y < 0.5) {
    //   rho = 0.8;
    //   v1 = 0.0;
    //   v2 = 0.0;
    //   p = 1.0;
    // } else if (x >= 0.5 && y < 0.5) {
    //   rho = 1.0;
    //   v1 = 0.0;
    //   v2 = 0.7276;
    //   p = 1.0;
    // }

    if (x >= 0.8 && y >= 0.8) {
      rho = 1.5;
      v1 = 0.0;
      v2 = 0.0;
      p = 1.5;
    } else if (x < 0.8 && y >= 0.8) {
      rho = 0.5323;
      v1 = 1.206;
      v2 = 0.0;
      p = 0.3;
    } else if (x < 0.8 && y < 0.8) {
      rho = 0.138;
      v1 = 1.206;
      v2 = 1.206;
      p = 0.029;
    } else if (x >= 0.8 && y < 0.8) {
      rho = 0.5323;
      v1 = 0.0;
      v2 = 1.206;
      p = 0.3;
    }

    return DGSEM::utils::prim_to_cons(std::array<T, 4>{rho, v1, v2, p},
                                      static_cast<T>(1.4));
  }
};

int main() {
  DGSEM::KokkosSession kokkos;
  {
    using value_type = double;
    using Eq = DGSEM::equations::CompressibleEuler2D<value_type>;
    using MyBasis = DGSEM::Basis::LobattoLegendreBasis<value_type, 5>;
    using SurfaceFlux = DGSEM::LaxFriedrichsFlux<Eq>;
    using VolumeFlux = DGSEM::VolumeIntegralShockCapturingHG<
        MyBasis, Eq, DGSEM::ChandrashekarFlux, DGSEM::LaxFriedrichsFlux,
        DGSEM::HGIndicator<MyBasis, Eq>>;
    using Mesh = DGSEM::StructuredMesh<value_type, 2>;

    DGSEM::BasisGuard<MyBasis> basis;

    auto boundaries =
        DGSEM::BoundarySet(DGSEM::OutflowBC{}, DGSEM::OutflowBC{},
                           DGSEM::OutflowBC{}, DGSEM::OutflowBC{});
    using Solver = DGSEM::StructuredSolver<Eq, MyBasis, VolumeFlux, SurfaceFlux,
                                           Mesh, decltype(boundaries)>;
    using Solution = DGSEM::Solution<Mesh, MyBasis, Eq>;

    std::size_t nx = 256;
    std::size_t ny = 256;
    value_type t_final = 0.8;

    std::array<value_type, 2> domain_left = {0.0, 0.0};
    std::array<value_type, 2> domain_right = {1.0, 1.0};

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
                    {false, false}};

    initializer.init_elements(n_cells, container);

    Solver solver(eq, mesh, container, boundaries);
    solver.set_indicator_parameters(0.5, 0.001, false);

    Solution sol(mesh);
    Riemann2DInitial<value_type> initial{};
    solver.initialize(initial, sol);

    using Analyzer =
        DGSEM::AnalyzerWrapper<MyBasis, Eq,
                               DGSEM::DivergenceChecker<value_type, Eq::NVARS>>;
    Analyzer analyzer;

    using DGSEM::PrintObserver;

    using VTUOutputObserver =
        DGSEM::VTUOutputObserver<value_type, MyBasis, Solution,
                                 decltype(container.node_coordinates), Eq>;

    using TimeIntegrator = DGSEM::SSPRK3<value_type, Solver, Mesh, Solution>;
    TimeIntegrator time_integrator(sol, mesh, t_final);

    // CFL, dt 计算
    const value_type cfl = 0.2;
    const value_type dx = (domain_right[0] - domain_left[0]) / nx;
    const value_type dy = (domain_right[1] - domain_left[1]) / ny;
    const value_type max_speed = 2.0;
    const value_type dt =
        cfl * std::min(dx, dy) / ((2.0 * MyBasis::NNodes - 1.0) * max_speed);

    time_integrator.add_observer(DGSEM::make_analysis_observer<MyBasis, Eq>(
        DGSEM::PointwiseAnalysisTag{}, analyzer, sol, n_cells,
        DGSEM::StopOnNaN<Eq>()));

    time_integrator.add_observer(std::make_unique<PrintObserver>(100));
    time_integrator.add_observer(std::make_unique<VTUOutputObserver>(
        "riemann2d_output", sol, container.node_coordinates, n_cells, 1000));

    // 用 observer 驱动积分
    time_integrator.solve(solver, sol, dt);

  }
  return 0;
}
