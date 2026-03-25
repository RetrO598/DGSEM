#pragma once

#include <Kokkos_Core.hpp>
#include <array>
#include <base/base.hpp>
#include <cstddef>
#include <decl/Kokkos_Declare_OPENMP.hpp>
#include <time_integrator/integrator_base.hpp>

namespace DGSEM {

template <class Equations>
struct parallel_ma3 {
  using traits = equations::EquationTraits<Equations>;
  using T = typename traits::value_type;
  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  using DataArray = typename solution_type_traits<T, NDIMS>::DataArray;

  parallel_ma3(DataArray u1_, DataArray u2_, DataArray u3_, DataArray du_, T a1,
               T a2, T a3, std::size_t n_dofs_)
      : u1(u1_), u2(u2_), u3(u3_), du(du_), a1(a1), a2(a2), a3(a3),
        n_dofs(n_dofs_) {}

  static void apply(DataArray u1_, DataArray u2_, DataArray u3_, DataArray du_,
                    T a1, T a2, T a3, std::size_t n_dofs_,
                    std::array<std::size_t, NDIMS> n_elems_)
    requires(NDIMS == 1)
  {
    parallel_ma3 functor(u1_, u2_, u3_, du_, a1, a2, a3, n_dofs_);

    Kokkos::parallel_for("parallel_ma", n_elems_[0], functor);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t &ielem) const
    requires(NDIMS == 1)
  {
    for (std::size_t inode = 0; inode < n_dofs; ++inode) {
      for (std::size_t ivar = 0; ivar < NVARS; ++ivar) {
        u1(ielem, inode, ivar) = a1 * u2(ielem, inode, ivar) +
                                 a2 * u3(ielem, inode, ivar) +
                                 a3 * du(ielem, inode, ivar);
      }
    }
  }

  DataArray u1;
  DataArray u2;
  DataArray u3;
  DataArray du;
  T a1, a2, a3;
  std::size_t n_dofs;
};

template <class Equations>
struct parallel_ma2 {
  using traits = equations::EquationTraits<Equations>;
  using T = typename traits::value_type;
  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  using DataArray = typename solution_type_traits<T, NDIMS>::DataArray;

  parallel_ma2(DataArray u1_, DataArray u2_, DataArray du_, T a1, T a2,
               std::size_t n_dofs_)
      : u1(u1_), u2(u2_), du(du_), a1(a1), a2(a2), n_dofs(n_dofs_) {}

  static void apply(DataArray u1_, DataArray u2_, DataArray du_, T a1, T a2,
                    std::size_t n_dofs_,
                    std::array<std::size_t, NDIMS> n_elems_)
    requires(NDIMS == 1)
  {
    parallel_ma2 functor(u1_, u2_, du_, a1, a2, n_dofs_);

    Kokkos::parallel_for("parallel_ma", n_elems_[0], functor);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t &ielem) const
    requires(NDIMS == 1)
  {
    for (std::size_t inode = 0; inode < n_dofs; ++inode) {
      for (std::size_t ivar = 0; ivar < NVARS; ++ivar) {
        u1(ielem, inode, ivar) =
            a1 * u2(ielem, inode, ivar) + a2 * du(ielem, inode, ivar);
      }
    }
  }

  DataArray u1;
  DataArray u2;
  DataArray du;
  T a1, a2;
  std::size_t n_dofs;
};

template <class T, class Solver, class Mesh, class Solution>
class SSPRK3 : public TimeIntegrator<T, Solver, Solution> {
public:
  using Equations = typename Solver::EquationType;
  explicit SSPRK3(const Solution &sol, const Mesh &mesh_)
      : tmp1(sol.clone_shape()), tmp2(sol.clone_shape()), mesh(mesh_){};

  void step(Solver &solver, Solution &sol, T dt) override {

    solver.calc_rhs(sol);

    parallel_ma2<Equations>::apply(tmp1.u_kokkos, sol.u_kokkos, sol.du_kokkos,
                                   1.0, dt, solver.get_ndofs(),
                                   mesh.get_num_cells());

    solver.calc_rhs(tmp1);

    parallel_ma3<Equations>::apply(
        tmp2.u_kokkos, sol.u_kokkos, tmp1.u_kokkos, tmp1.du_kokkos, 3.0 / 4.0,
        1.0 / 4.0, 1.0 / 4.0 * dt, solver.get_ndofs(), mesh.get_num_cells());

    solver.calc_rhs(tmp2);

    parallel_ma3<Equations>::apply(
        sol.u_kokkos, sol.u_kokkos, tmp2.u_kokkos, tmp2.du_kokkos, 1.0 / 3.0,
        2.0 / 3.0, 2.0 / 3.0 * dt, solver.get_ndofs(), mesh.get_num_cells());
  }

private:
  Solution tmp1;
  Solution tmp2;
  const Mesh &mesh;
};
} // namespace DGSEM