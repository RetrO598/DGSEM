#pragma once

#include <base/base.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <tuple>
#include <type_traits>
#include <utils/local_dof.hpp>

namespace DGSEM {

template <std::size_t NDIMS, class BC, class Equations, class SurfaceFlux,
          class Mesh, class T, class ArrayU, class ElementData, class ArrayFlux>
struct BCDispatcher;

template <class BC, class Equations, class SurfaceFlux, class Mesh, class T,
          class ArrayU, class ElementData, class ArrayFlux>
struct BCDispatcher<1, BC, Equations, SurfaceFlux, Mesh, T, ArrayU, ElementData,
                    ArrayFlux> {
  static void dispatch(const BC& bc, const Mesh& mesh, const Equations& eq,
                       const ArrayU& u, const ElementData& element_data,
                       ArrayFlux& surface_flux, std::size_t face_id, T time) {
    // 1D case: No parallel_for needed for many elements,
    // but we use a single-threaded kernel to access device data.
    Kokkos::parallel_for(
        "BC_Apply_1D", 1, KOKKOS_LAMBDA(int) {
          bc.template apply_device<Equations, SurfaceFlux, Mesh, T, 1>(
              mesh, eq, u, element_data, surface_flux, face_id, time);
        });
  }
};

template <class BC, class Equations, class SurfaceFlux, class Mesh, class T,
          class ArrayU, class ElementData, class ArrayFlux>
struct BCDispatcher<2, BC, Equations, SurfaceFlux, Mesh, T, ArrayU, ElementData,
                    ArrayFlux> {
  static void dispatch(const BC& bc, const Mesh& mesh, const Equations& eq,
                       const ArrayU& u, const ElementData& element_data,
                       ArrayFlux& surface_flux, std::size_t face_id, T time) {
    auto n_cells = mesh.get_num_cells();
    std::size_t range = (face_id < 2) ? n_cells[1] : n_cells[0];

    Kokkos::parallel_for(
        "BC_Apply_2D", range, KOKKOS_LAMBDA(int i) {
          bc.template apply_device<Equations, SurfaceFlux, Mesh, T, 2>(
              mesh, eq, u, element_data, surface_flux, face_id, time, i);
        });
  }
};

template <class... FaceBCs>
struct BoundarySet {
  static constexpr std::size_t NFACES = sizeof...(FaceBCs);

  std::tuple<FaceBCs...> faces_;

  BoundarySet(FaceBCs... bcs) : faces_(std::move(bcs)...) {}

  template <std::size_t I = 0, typename F>
  void tuple_switch(std::size_t idx, F&& f) const {
    if constexpr (I < NFACES) {
      if (I == idx) {
        f(std::get<I>(faces_));
      } else {
        tuple_switch<I + 1>(idx, std::forward<F>(f));
      }
    } else {
      // throw std::out_of_range("BoundarySet::apply: face_id out of range");
    }
  }

  // Kokkos apply
  template <class Equations, class SurfaceFlux, class Mesh, class T,
            std::size_t NDIMS, class ArrayU, class ElementData, class ArrayFlux>
  void apply(const Mesh& mesh, const Equations& eq, const ArrayU& u,
             const ElementData& element_data, ArrayFlux& surface_flux,
             std::size_t face_id, T time) const {
    if constexpr (NFACES > 0) {
      tuple_switch(face_id, [&](const auto& bc) {
        BCDispatcher<NDIMS, std::decay_t<decltype(bc)>, Equations, SurfaceFlux,
                     Mesh, T, ArrayU, ElementData,
                     ArrayFlux>::dispatch(bc, mesh, eq, u, element_data,
                                          surface_flux, face_id, time);
      });
    }
  }
};
} // namespace DGSEM
