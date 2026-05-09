#pragma once

#include <Kokkos_Core.hpp>
#include <array>
#include <base/base.hpp>
#include <cstddef>
#include <time_integrator/integrator_base.hpp>
#include <utils/cell_iteration.hpp>

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
                    std::array<std::size_t, NDIMS> n_elems_) {
    parallel_ma3 functor(u1_, u2_, u3_, du_, a1, a2, a3, n_dofs_);
    DGSEM::utils::parallel_for_cells<NDIMS>("parallel_ma3", n_elems_, functor);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& ielem) const
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

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& ielem,
                                         const std::size_t& jelem) const
    requires(NDIMS == 2)
  {
    for (std::size_t inode = 0; inode < n_dofs; ++inode) {
      for (std::size_t ivar = 0; ivar < NVARS; ++ivar) {
        u1(ielem, jelem, inode, ivar) = a1 * u2(ielem, jelem, inode, ivar) +
                                        a2 * u3(ielem, jelem, inode, ivar) +
                                        a3 * du(ielem, jelem, inode, ivar);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& ielem,
                                         const std::size_t& jelem,
                                         const std::size_t& kelem) const
    requires(NDIMS == 3)
  {
    for (std::size_t inode = 0; inode < n_dofs; ++inode) {
      for (std::size_t ivar = 0; ivar < NVARS; ++ivar) {
        u1(ielem, jelem, kelem, inode, ivar) =
            a1 * u2(ielem, jelem, kelem, inode, ivar) +
            a2 * u3(ielem, jelem, kelem, inode, ivar) +
            a3 * du(ielem, jelem, kelem, inode, ivar);
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
                    std::array<std::size_t, NDIMS> n_elems_) {
    parallel_ma2 functor(u1_, u2_, du_, a1, a2, n_dofs_);
    DGSEM::utils::parallel_for_cells<NDIMS>("parallel_ma2", n_elems_, functor);
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& ielem) const
    requires(NDIMS == 1)
  {
    for (std::size_t inode = 0; inode < n_dofs; ++inode) {
      for (std::size_t ivar = 0; ivar < NVARS; ++ivar) {
        u1(ielem, inode, ivar) =
            a1 * u2(ielem, inode, ivar) + a2 * du(ielem, inode, ivar);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& ielem,
                                         const std::size_t& jelem) const
    requires(NDIMS == 2)
  {
    for (std::size_t inode = 0; inode < n_dofs; ++inode) {
      for (std::size_t ivar = 0; ivar < NVARS; ++ivar) {
        u1(ielem, jelem, inode, ivar) = a1 * u2(ielem, jelem, inode, ivar) +
                                        a2 * du(ielem, jelem, inode, ivar);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t& ielem,
                                         const std::size_t& jelem,
                                         const std::size_t& kelem) const
    requires(NDIMS == 3)
  {
    for (std::size_t inode = 0; inode < n_dofs; ++inode) {
      for (std::size_t ivar = 0; ivar < NVARS; ++ivar) {
        u1(ielem, jelem, kelem, inode, ivar) =
            a1 * u2(ielem, jelem, kelem, inode, ivar) +
            a2 * du(ielem, jelem, kelem, inode, ivar);
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
  using base = TimeIntegrator<T, Solver, Solution>;

  explicit SSPRK3(const Solution& sol, const Mesh& mesh_)
      : base(), tmp1(sol.clone_shape()), tmp2(sol.clone_shape()),
        mesh(mesh_) {};

  explicit SSPRK3(const Solution& sol, const Mesh& mesh_, T final_time_)
      : base(final_time_), tmp1(sol.clone_shape()), tmp2(sol.clone_shape()),
        mesh(mesh_) {};

  void step(Solver& solver, Solution& sol, T dt) override {
    solver.calc_rhs(sol, this->time);

    parallel_ma2<Equations>::apply(tmp1.state(), sol.state(), sol.rhs(),
                                   static_cast<T>(1.0), dt, solver.get_ndofs(),
                                   mesh.get_num_cells());

    solver.calc_rhs(tmp1, this->time + dt);

    parallel_ma3<Equations>::apply(tmp2.state(), sol.state(), tmp1.state(),
                                   tmp1.rhs(), static_cast<T>(3.0 / 4.0),
                                   static_cast<T>(1.0 / 4.0),
                                   static_cast<T>(1.0 / 4.0) * dt,
                                   solver.get_ndofs(), mesh.get_num_cells());

    solver.calc_rhs(tmp2, this->time + static_cast<T>(0.5) * dt);

    parallel_ma3<Equations>::apply(sol.state(), sol.state(), tmp2.state(),
                                   tmp2.rhs(), static_cast<T>(1.0 / 3.0),
                                   static_cast<T>(2.0 / 3.0),
                                   static_cast<T>(2.0 / 3.0) * dt,
                                   solver.get_ndofs(), mesh.get_num_cells());
  }

private:
  Solution tmp1;
  Solution tmp2;
  const Mesh& mesh;
};
} // namespace DGSEM
