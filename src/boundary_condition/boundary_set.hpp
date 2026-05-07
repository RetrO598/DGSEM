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

template <class BC, class Equations, class Mesh, class T, std::size_t NDIMS,
          class ArrayU, class ElementData, class ArrayFlux>
KOKKOS_INLINE_FUNCTION void
apply_gradient_bc_device_if_available(const BC& bc, const Mesh& mesh,
                                      const Equations& eq, const ArrayU& u,
                                      const ElementData& element_data,
                                      ArrayFlux& surface_flux,
                                      std::size_t face_id, T time,
                                      int index) {
  if constexpr (requires {
                  bc.template apply_gradient_device<Equations, Mesh, T, NDIMS>(
                      mesh, eq, u, element_data, surface_flux, face_id, time,
                      index);
                }) {
    bc.template apply_gradient_device<Equations, Mesh, T, NDIMS>(
        mesh, eq, u, element_data, surface_flux, face_id, time, index);
  }
}

template <class BC, class Equations, class Mesh, class T, std::size_t NDIMS,
          class ArrayU, class ElementData, class ArrayViscousFlux,
          class ArrayFlux>
KOKKOS_INLINE_FUNCTION void apply_viscous_bc_device_if_available(
    const BC& bc, const Mesh& mesh, const Equations& eq, const ArrayU& u,
    const ElementData& element_data, const ArrayViscousFlux& viscous_flux,
    ArrayFlux& surface_flux, std::size_t face_id, T time, int index) {
  if constexpr (requires {
                  bc.template apply_viscous_device<Equations, Mesh, T, NDIMS>(
                      mesh, eq, u, element_data, viscous_flux, surface_flux,
                      face_id, time, index);
                }) {
    bc.template apply_viscous_device<Equations, Mesh, T, NDIMS>(
        mesh, eq, u, element_data, viscous_flux, surface_flux, face_id, time,
        index);
  }
}

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

template <class BC, class Equations, class SurfaceFlux, class Mesh, class T,
          class ArrayU, class ElementData, class ArrayFlux>
struct BCDispatcher<3, BC, Equations, SurfaceFlux, Mesh, T, ArrayU, ElementData,
                    ArrayFlux> {
  static void dispatch(const BC& bc, const Mesh& mesh, const Equations& eq,
                       const ArrayU& u, const ElementData& element_data,
                       ArrayFlux& surface_flux, std::size_t face_id, T time) {
    auto n_cells = mesh.get_num_cells();
    std::size_t range = 0;
    if (face_id < 2) {
      range = n_cells[1] * n_cells[2];
    } else if (face_id < 4) {
      range = n_cells[0] * n_cells[2];
    } else {
      range = n_cells[0] * n_cells[1];
    }

    Kokkos::parallel_for(
        "BC_Apply_3D", range, KOKKOS_LAMBDA(int i) {
          bc.template apply_device<Equations, SurfaceFlux, Mesh, T, 3>(
              mesh, eq, u, element_data, surface_flux, face_id, time, i);
        });
  }
};

template <std::size_t NDIMS, class BC, class Equations, class Mesh, class T,
          class ArrayU, class ElementData, class ArrayFlux>
struct GradientBCDispatcher {
  static void dispatch(const BC& bc, const Mesh& mesh, const Equations& eq,
                       const ArrayU& u, const ElementData& element_data,
                       ArrayFlux& surface_flux, std::size_t face_id, T time) {
    std::size_t range = 1;
    if constexpr (NDIMS == 2) {
      auto n_cells = mesh.get_num_cells();
      range = (face_id < 2) ? n_cells[1] : n_cells[0];
    } else if constexpr (NDIMS == 3) {
      auto n_cells = mesh.get_num_cells();
      if (face_id < 2) {
        range = n_cells[1] * n_cells[2];
      } else if (face_id < 4) {
        range = n_cells[0] * n_cells[2];
      } else {
        range = n_cells[0] * n_cells[1];
      }
    }

    Kokkos::parallel_for(
        "Gradient_BC_Apply", range, KOKKOS_LAMBDA(int i) {
          apply_gradient_bc_device_if_available<BC, Equations, Mesh, T, NDIMS>(
              bc, mesh, eq, u, element_data, surface_flux, face_id, time, i);
        });
  }
};

template <std::size_t NDIMS, class BC, class Equations, class Mesh, class T,
          class ArrayU, class ElementData, class ArrayViscousFlux,
          class ArrayFlux>
struct ViscousBCDispatcher {
  static void dispatch(const BC& bc, const Mesh& mesh, const Equations& eq,
                       const ArrayU& u, const ElementData& element_data,
                       const ArrayViscousFlux& viscous_flux,
                       ArrayFlux& surface_flux, std::size_t face_id, T time) {
    std::size_t range = 1;
    if constexpr (NDIMS == 2) {
      auto n_cells = mesh.get_num_cells();
      range = (face_id < 2) ? n_cells[1] : n_cells[0];
    } else if constexpr (NDIMS == 3) {
      auto n_cells = mesh.get_num_cells();
      if (face_id < 2) {
        range = n_cells[1] * n_cells[2];
      } else if (face_id < 4) {
        range = n_cells[0] * n_cells[2];
      } else {
        range = n_cells[0] * n_cells[1];
      }
    }

    Kokkos::parallel_for(
        "Viscous_BC_Apply", range, KOKKOS_LAMBDA(int i) {
          apply_viscous_bc_device_if_available<BC, Equations, Mesh, T, NDIMS>(
              bc, mesh, eq, u, element_data, viscous_flux, surface_flux,
              face_id, time, i);
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

  template <class Equations, class Mesh, class T, std::size_t NDIMS,
            class ArrayU, class ElementData, class ArrayFlux>
  void apply_gradient(const Mesh& mesh, const Equations& eq, const ArrayU& u,
                      const ElementData& element_data, ArrayFlux& surface_flux,
                      std::size_t face_id, T time) const {
    if constexpr (NFACES > 0) {
      tuple_switch(face_id, [&](const auto& bc) {
        GradientBCDispatcher<NDIMS, std::decay_t<decltype(bc)>, Equations,
                             Mesh, T, ArrayU, ElementData,
                             ArrayFlux>::dispatch(bc, mesh, eq, u,
                                                  element_data, surface_flux,
                                                  face_id, time);
      });
    }
  }

  template <class Equations, class Mesh, class T, std::size_t NDIMS,
            class ArrayU, class ElementData, class ArrayViscousFlux,
            class ArrayFlux>
  void apply_viscous(const Mesh& mesh, const Equations& eq, const ArrayU& u,
                     const ElementData& element_data,
                     const ArrayViscousFlux& viscous_flux,
                     ArrayFlux& surface_flux, std::size_t face_id,
                     T time) const {
    if constexpr (NFACES > 0) {
      tuple_switch(face_id, [&](const auto& bc) {
        ViscousBCDispatcher<NDIMS, std::decay_t<decltype(bc)>, Equations, Mesh,
                            T, ArrayU, ElementData, ArrayViscousFlux,
                            ArrayFlux>::dispatch(bc, mesh, eq, u,
                                                 element_data, viscous_flux,
                                                 surface_flux, face_id, time);
      });
    }
  }
};
} // namespace DGSEM
