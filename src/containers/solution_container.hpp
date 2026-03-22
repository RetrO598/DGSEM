#include "utils/kokkos_helper.hpp"
#include <Kokkos_Core.hpp>
#include <base/base.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <utils/utils.hpp>
#include <xtensor.hpp>
#include <xtensor/core/xtensor_forward.hpp>

namespace DGSEM {

namespace detail {
template <class T, std::size_t NVARS, std::size_t NDIMS>
struct SolutionInitializer;

template <class T, std::size_t NVARS>
struct SolutionInitializer<T, NVARS, 1> {

  using ndarray = xt::xarray<T>;

  using DataArray = solution_type_traits<T, 1>::DataArray;
  using DataArrayHost = solution_type_traits<T, 1>::DataArrayHost;

  inline constexpr static void initialize_u(std::size_t total_elements,
                                            std::size_t nnodes, ndarray &u,
                                            DataArray &u_kokkos) {
    u.resize({total_elements, nnodes, NVARS});
    u.fill(0.0);

    Kokkos::realloc(u_kokkos, total_elements, nnodes, NVARS);
    Kokkos::deep_copy(u_kokkos, 0.0);
  }

  inline constexpr static void initialize_du(std::size_t total_elements,
                                             std::size_t nnodes, ndarray &du,
                                             DataArray &du_kokkos) {
    du.resize({total_elements, nnodes, NVARS});
    du.fill(0.0);

    Kokkos::realloc(du_kokkos, total_elements, nnodes, NVARS);
    Kokkos::deep_copy(du_kokkos, 0.0);
  }

  inline constexpr static void
  initialize_surface_flux_value(std::size_t total_elements, std::size_t nnodes,
                                ndarray &surface_flux_value,
                                DataArray &surface_kokkos) {
    surface_flux_value.resize({total_elements, 2, NVARS});
    surface_flux_value.fill(0.0);

    Kokkos::realloc(surface_kokkos, total_elements, 2, NVARS);
    Kokkos::deep_copy(surface_kokkos, 0.0);
  }
};
} // namespace detail

template <class Mesh, class Basis, class Equations>
struct Solution {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;
  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  using ndarray = xt::xarray<value_type>;

  using DataArray = solution_type_traits<value_type, NDIMS>::DataArray;
  using DataArrayHost = solution_type_traits<value_type, NDIMS>::DataArrayHost;

  Solution(const Mesh &mesh) {
    std::size_t total_elements = mesh.get_nelem();
    detail::SolutionInitializer<value_type, NVARS, NDIMS>::initialize_u(
        total_elements, Basis::NNodes, u, u_kokkos);

    detail::SolutionInitializer<value_type, NVARS, NDIMS>::initialize_du(
        total_elements, Basis::NNodes, du, du_kokkos);

    detail::SolutionInitializer<value_type, NVARS, NDIMS>::
        initialize_surface_flux_value(total_elements, Basis::NNodes,
                                      surface_flux_value,
                                      surface_flux_value_kokkos);
  }

  Solution() = default;

  Solution clone_shape() const {
    Solution tmp;
    tmp.u.resize(u.shape());
    tmp.du.resize(du.shape());
    tmp.surface_flux_value.resize(surface_flux_value.shape());
    tmp.u.fill(0.0);
    tmp.du.fill(0.0);
    tmp.surface_flux_value.fill(0.0);

    clone_view_shape(tmp.u_kokkos, u_kokkos);
    clone_view_shape(tmp.du_kokkos, du_kokkos);
    clone_view_shape(tmp.surface_flux_value_kokkos, surface_flux_value_kokkos);

    return tmp;
  }

  void check_solution() {
    compare_view_xtensor(u_kokkos, u);
    compare_view_xtensor(du_kokkos, du);
    compare_view_xtensor(surface_flux_value_kokkos, surface_flux_value);
  }

  ndarray u;
  ndarray du;
  ndarray surface_flux_value;

  DataArray u_kokkos;
  DataArray du_kokkos;
  DataArray surface_flux_value_kokkos;
};
} // namespace DGSEM