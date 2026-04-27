#pragma once

#include <algorithm>
#include <array>
#include <base/base.hpp>
#include <cmath>
#include <cstddef>
#include <equations/equations.hpp>
#include <limits>
#include <utils/local_dof.hpp>

namespace DGSEM {

namespace detail {
template <class T, std::size_t NVARS, std::size_t NDIMS>
struct VolumeIntegralWeak;

template <class T, std::size_t NVARS>
struct VolumeIntegralWeak<T, NVARS, 1> {
  template <class BasisData, class Equations, class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION static void
  weak_form_kernel(std::size_t ielem, const BasisData& basis,
                   const Equations& eq, const ArrayU& u, ArrayDu& du) {
    for (std::size_t inode = 0; inode < BasisData::NNodes; ++inode) {
      std::array<T, NVARS> u_node;
      std::array<T, NVARS> flux;
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_node[var] = u(ielem, inode, var);
      }
      flux = eq.flux(u_node, 0);

      for (std::size_t iinode = 0; iinode < BasisData::NNodes; ++iinode) {
        for (std::size_t var = 0; var < NVARS; ++var) {
          du(ielem, iinode, var) =
              du(ielem, iinode, var) +
              basis.derivative_dhat(iinode, inode) * flux[var];
        }
      }
    }
  }
};

template <class T, std::size_t NVARS>
struct VolumeIntegralWeak<T, NVARS, 2> {
  template <class BasisData, class Equations, class ArrayU, class ArrayDu,
            class MetricArray>
  KOKKOS_INLINE_FUNCTION static void
  weak_form_kernel(std::size_t ielem, std::size_t jelem, const BasisData& basis,
                   const MetricArray& contravariant_vectors,
                   const Equations& eq, const ArrayU& u, ArrayDu& du) {
    for (std::size_t jnode = 0; jnode < BasisData::NNodes; ++jnode) {
      for (std::size_t inode = 0; inode < BasisData::NNodes; ++inode) {
        const std::size_t dof =
            DGSEM::utils::local_dof<BasisData::NNodes>(inode, jnode);
        std::array<T, NVARS> u_node{};
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_node[var] = u(ielem, jelem, dof, var);
        }

        auto flux_x = eq.flux(u_node, 0);
        auto flux_y = eq.flux(u_node, 1);
        std::array<T, NVARS> contravariant_flux_xi{};
        std::array<T, NVARS> contravariant_flux_eta{};

        const T ja11 = contravariant_vectors(ielem, jelem, dof, 0, 0);
        const T ja12 = contravariant_vectors(ielem, jelem, dof, 0, 1);
        const T ja21 = contravariant_vectors(ielem, jelem, dof, 1, 0);
        const T ja22 = contravariant_vectors(ielem, jelem, dof, 1, 1);

        for (std::size_t var = 0; var < NVARS; ++var) {
          contravariant_flux_xi[var] = ja11 * flux_x[var] + ja12 * flux_y[var];
          contravariant_flux_eta[var] = ja21 * flux_x[var] + ja22 * flux_y[var];
        }

        for (std::size_t iinode = 0; iinode < BasisData::NNodes; ++iinode) {
          const std::size_t out_dof =
              DGSEM::utils::local_dof<BasisData::NNodes>(iinode, jnode);
          for (std::size_t var = 0; var < NVARS; ++var) {
            du(ielem, jelem, out_dof, var) +=
                basis.derivative_dhat(iinode, inode) *
                contravariant_flux_xi[var];
          }
        }

        for (std::size_t jjnode = 0; jjnode < BasisData::NNodes; ++jjnode) {
          const std::size_t out_dof =
              DGSEM::utils::local_dof<BasisData::NNodes>(inode, jjnode);
          for (std::size_t var = 0; var < NVARS; ++var) {
            du(ielem, jelem, out_dof, var) +=
                basis.derivative_dhat(jjnode, jnode) *
                contravariant_flux_eta[var];
          }
        }
      }
    }
  }
};

template <class T, std::size_t NVARS>
struct VolumeIntegralWeak<T, NVARS, 3> {
  template <class BasisData, class Equations, class ArrayU, class ArrayDu,
            class MetricArray>
  KOKKOS_INLINE_FUNCTION static void
  weak_form_kernel(std::size_t ielem, std::size_t jelem, std::size_t kelem,
                   const BasisData& basis,
                   const MetricArray& contravariant_vectors,
                   const Equations& eq, const ArrayU& u, ArrayDu& du) {
    for (std::size_t knode = 0; knode < BasisData::NNodes; ++knode) {
      for (std::size_t jnode = 0; jnode < BasisData::NNodes; ++jnode) {
        for (std::size_t inode = 0; inode < BasisData::NNodes; ++inode) {
          const std::size_t dof =
              DGSEM::utils::local_dof<BasisData::NNodes>(inode, jnode, knode);
          std::array<T, NVARS> u_node{};
          for (std::size_t var = 0; var < NVARS; ++var) {
            u_node[var] = u(ielem, jelem, kelem, dof, var);
          }

          const auto flux_x = eq.flux(u_node, 0);
          const auto flux_y = eq.flux(u_node, 1);
          const auto flux_z = eq.flux(u_node, 2);
          std::array<T, NVARS> contravariant_flux_xi{};
          std::array<T, NVARS> contravariant_flux_eta{};
          std::array<T, NVARS> contravariant_flux_zeta{};

          const T ja11 = contravariant_vectors(ielem, jelem, kelem, dof, 0, 0);
          const T ja12 = contravariant_vectors(ielem, jelem, kelem, dof, 0, 1);
          const T ja13 = contravariant_vectors(ielem, jelem, kelem, dof, 0, 2);
          const T ja21 = contravariant_vectors(ielem, jelem, kelem, dof, 1, 0);
          const T ja22 = contravariant_vectors(ielem, jelem, kelem, dof, 1, 1);
          const T ja23 = contravariant_vectors(ielem, jelem, kelem, dof, 1, 2);
          const T ja31 = contravariant_vectors(ielem, jelem, kelem, dof, 2, 0);
          const T ja32 = contravariant_vectors(ielem, jelem, kelem, dof, 2, 1);
          const T ja33 = contravariant_vectors(ielem, jelem, kelem, dof, 2, 2);

          for (std::size_t var = 0; var < NVARS; ++var) {
            contravariant_flux_xi[var] =
                ja11 * flux_x[var] + ja12 * flux_y[var] + ja13 * flux_z[var];
            contravariant_flux_eta[var] =
                ja21 * flux_x[var] + ja22 * flux_y[var] + ja23 * flux_z[var];
            contravariant_flux_zeta[var] =
                ja31 * flux_x[var] + ja32 * flux_y[var] + ja33 * flux_z[var];
          }

          for (std::size_t iinode = 0; iinode < BasisData::NNodes; ++iinode) {
            const std::size_t out_dof =
                DGSEM::utils::local_dof<BasisData::NNodes>(iinode, jnode,
                                                           knode);
            for (std::size_t var = 0; var < NVARS; ++var) {
              du(ielem, jelem, kelem, out_dof, var) +=
                  basis.derivative_dhat(iinode, inode) *
                  contravariant_flux_xi[var];
            }
          }

          for (std::size_t jjnode = 0; jjnode < BasisData::NNodes; ++jjnode) {
            const std::size_t out_dof =
                DGSEM::utils::local_dof<BasisData::NNodes>(inode, jjnode,
                                                           knode);
            for (std::size_t var = 0; var < NVARS; ++var) {
              du(ielem, jelem, kelem, out_dof, var) +=
                  basis.derivative_dhat(jjnode, jnode) *
                  contravariant_flux_eta[var];
            }
          }

          for (std::size_t kknode = 0; kknode < BasisData::NNodes; ++kknode) {
            const std::size_t out_dof =
                DGSEM::utils::local_dof<BasisData::NNodes>(inode, jnode,
                                                           kknode);
            for (std::size_t var = 0; var < NVARS; ++var) {
              du(ielem, jelem, kelem, out_dof, var) +=
                  basis.derivative_dhat(kknode, knode) *
                  contravariant_flux_zeta[var];
            }
          }
        }
      }
    }
  }
};

template <class T, std::size_t NVARS, class NumericFlux, std::size_t NDIMS>
struct VolumeIntegralSplit;

template <class T, std::size_t NVARS, class NumericFlux>
struct VolumeIntegralSplit<T, NVARS, NumericFlux, 1> {
  template <class BasisData, class Equations, class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION static void
  split_form_kernel(std::size_t ielem, const BasisData& basis,
                    const Equations& eq, const ArrayU& u, ArrayDu& du,
                    T alpha = static_cast<T>(1.0)) {
    for (std::size_t inode = 0; inode < BasisData::NNodes; ++inode) {
      std::array<T, NVARS> u_node;
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_node[var] = u(ielem, inode, var);
      }

      for (std::size_t jnode = inode + 1; jnode < BasisData::NNodes; ++jnode) {
        std::array<T, NVARS> u_node_j;
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_node_j[var] = u(ielem, jnode, var);
        }

        auto flux_ij = NumericFlux::numerical_flux(eq, u_node, u_node_j);

        // Reduce the computation by exploiting the symmetry of the flux. Note
        // the diagonal terms of Dsplit matrix is zero.
        for (std::size_t var = 0; var < NVARS; ++var) {
          du(ielem, inode, var) =
              du(ielem, inode, var) +
              alpha * basis.derivative_split(inode, jnode) * flux_ij[var];
          du(ielem, jnode, var) =
              du(ielem, jnode, var) +
              alpha * basis.derivative_split(jnode, inode) * flux_ij[var];
        }
      }
    }
  }
};

template <class T, std::size_t NVARS, class NumericFlux>
struct VolumeIntegralSplit<T, NVARS, NumericFlux, 2> {
  template <class BasisData, class Equations, class ArrayU, class ArrayDu,
            class MetricArray>
  KOKKOS_INLINE_FUNCTION static void split_form_kernel(
      std::size_t ielem, std::size_t jelem, const BasisData& basis,
      const MetricArray& contravariant_vectors, const Equations& eq,
      const ArrayU& u, ArrayDu& du, T alpha = static_cast<T>(1.0)) {
    for (std::size_t jnode = 0; jnode < BasisData::NNodes; ++jnode) {
      for (std::size_t inode = 0; inode < BasisData::NNodes; ++inode) {
        const std::size_t dof_i =
            DGSEM::utils::local_dof<BasisData::NNodes>(inode, jnode);
        std::array<T, NVARS> u_node{};
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_node[var] = u(ielem, jelem, dof_i, var);
        }

        for (std::size_t knode = inode + 1; knode < BasisData::NNodes;
             ++knode) {
          const std::size_t dof_k =
              DGSEM::utils::local_dof<BasisData::NNodes>(knode, jnode);
          std::array<T, NVARS> u_node_k{};
          for (std::size_t var = 0; var < NVARS; ++var) {
            u_node_k[var] = u(ielem, jelem, dof_k, var);
          }

          const std::array<T, 2> normal = {
              (T)0.5 * (contravariant_vectors(ielem, jelem, dof_i, 0, 0) +
                        contravariant_vectors(ielem, jelem, dof_k, 0, 0)),
              (T)0.5 * (contravariant_vectors(ielem, jelem, dof_i, 0, 1) +
                        contravariant_vectors(ielem, jelem, dof_k, 0, 1))};
          const auto flux_ik =
              NumericFlux::numerical_flux(eq, u_node, u_node_k, normal);

          // Reduce the computation by exploiting the symmetry of the flux. Note
          // the diagonal terms of Dsplit matrix is zero.
          for (std::size_t var = 0; var < NVARS; ++var) {
            du(ielem, jelem, dof_i, var) +=
                alpha * basis.derivative_split(inode, knode) * flux_ik[var];
            du(ielem, jelem, dof_k, var) +=
                alpha * basis.derivative_split(knode, inode) * flux_ik[var];
          }
        }

        for (std::size_t lnode = jnode + 1; lnode < BasisData::NNodes;
             ++lnode) {
          const std::size_t dof_l =
              DGSEM::utils::local_dof<BasisData::NNodes>(inode, lnode);
          std::array<T, NVARS> u_node_l{};
          for (std::size_t var = 0; var < NVARS; ++var) {
            u_node_l[var] = u(ielem, jelem, dof_l, var);
          }

          const std::array<T, 2> normal = {
              (T)0.5 * (contravariant_vectors(ielem, jelem, dof_i, 1, 0) +
                        contravariant_vectors(ielem, jelem, dof_l, 1, 0)),
              (T)0.5 * (contravariant_vectors(ielem, jelem, dof_i, 1, 1) +
                        contravariant_vectors(ielem, jelem, dof_l, 1, 1))};
          const auto flux_jl =
              NumericFlux::numerical_flux(eq, u_node, u_node_l, normal);

          // Reduce the computation by exploiting the symmetry of the flux. Note
          // the diagonal terms of Dsplit matrix is zero.
          for (std::size_t var = 0; var < NVARS; ++var) {
            du(ielem, jelem, dof_i, var) +=
                alpha * basis.derivative_split(jnode, lnode) * flux_jl[var];
            du(ielem, jelem, dof_l, var) +=
                alpha * basis.derivative_split(lnode, jnode) * flux_jl[var];
          }
        }
      }
    }
  }
};

template <class T, std::size_t NVARS, class NumericFlux>
struct VolumeIntegralSplit<T, NVARS, NumericFlux, 3> {
  template <class BasisData, class Equations, class ArrayU, class ArrayDu,
            class MetricArray>
  KOKKOS_INLINE_FUNCTION static void
  split_form_kernel(std::size_t ielem, std::size_t jelem, std::size_t kelem,
                    const BasisData& basis,
                    const MetricArray& contravariant_vectors,
                    const Equations& eq, const ArrayU& u, ArrayDu& du,
                    T alpha = static_cast<T>(1.0)) {

    for (std::size_t knode = 0; knode < BasisData::NNodes; ++knode) {
      for (std::size_t jnode = 0; jnode < BasisData::NNodes; ++jnode) {
        for (std::size_t inode = 0; inode < BasisData::NNodes; ++inode) {
          const std::size_t dof_i =
              DGSEM::utils::local_dof<BasisData::NNodes>(inode, jnode, knode);
          std::array<T, NVARS> u_node{};
          for (std::size_t var = 0; var < NVARS; ++var) {
            u_node[var] = u(ielem, jelem, kelem, dof_i, var);
          }

          for (std::size_t lnode = inode + 1; lnode < BasisData::NNodes;
               ++lnode) {
            const std::size_t dof_l =
                DGSEM::utils::local_dof<BasisData::NNodes>(lnode, jnode, knode);
            std::array<T, NVARS> u_node_l{};
            for (std::size_t var = 0; var < NVARS; ++var) {
              u_node_l[var] = u(ielem, jelem, kelem, dof_l, var);
            }

            const std::array<T, 3> normal = {
                static_cast<T>(0.5) *
                    (contravariant_vectors(ielem, jelem, kelem, dof_i, 0, 0) +
                     contravariant_vectors(ielem, jelem, kelem, dof_l, 0, 0)),
                static_cast<T>(0.5) *
                    (contravariant_vectors(ielem, jelem, kelem, dof_i, 0, 1) +
                     contravariant_vectors(ielem, jelem, kelem, dof_l, 0, 1)),
                static_cast<T>(0.5) *
                    (contravariant_vectors(ielem, jelem, kelem, dof_i, 0, 2) +
                     contravariant_vectors(ielem, jelem, kelem, dof_l, 0, 2))};
            const auto flux_il =
                NumericFlux::numerical_flux(eq, u_node, u_node_l, normal);

            for (std::size_t var = 0; var < NVARS; ++var) {
              du(ielem, jelem, kelem, dof_i, var) +=
                  alpha * basis.derivative_split(inode, lnode) * flux_il[var];
              du(ielem, jelem, kelem, dof_l, var) +=
                  alpha * basis.derivative_split(lnode, inode) * flux_il[var];
            }
          }

          for (std::size_t mnode = jnode + 1; mnode < BasisData::NNodes;
               ++mnode) {
            const std::size_t dof_m =
                DGSEM::utils::local_dof<BasisData::NNodes>(inode, mnode, knode);
            std::array<T, NVARS> u_node_m{};
            for (std::size_t var = 0; var < NVARS; ++var) {
              u_node_m[var] = u(ielem, jelem, kelem, dof_m, var);
            }

            const std::array<T, 3> normal = {
                static_cast<T>(0.5) *
                    (contravariant_vectors(ielem, jelem, kelem, dof_i, 1, 0) +
                     contravariant_vectors(ielem, jelem, kelem, dof_m, 1, 0)),
                static_cast<T>(0.5) *
                    (contravariant_vectors(ielem, jelem, kelem, dof_i, 1, 1) +
                     contravariant_vectors(ielem, jelem, kelem, dof_m, 1, 1)),
                static_cast<T>(0.5) *
                    (contravariant_vectors(ielem, jelem, kelem, dof_i, 1, 2) +
                     contravariant_vectors(ielem, jelem, kelem, dof_m, 1, 2))};
            const auto flux_im =
                NumericFlux::numerical_flux(eq, u_node, u_node_m, normal);

            for (std::size_t var = 0; var < NVARS; ++var) {
              du(ielem, jelem, kelem, dof_i, var) +=
                  alpha * basis.derivative_split(jnode, mnode) * flux_im[var];
              du(ielem, jelem, kelem, dof_m, var) +=
                  alpha * basis.derivative_split(mnode, jnode) * flux_im[var];
            }
          }

          for (std::size_t nnode = knode + 1; nnode < BasisData::NNodes;
               ++nnode) {
            const std::size_t dof_n =
                DGSEM::utils::local_dof<BasisData::NNodes>(inode, jnode, nnode);
            std::array<T, NVARS> u_node_n{};
            for (std::size_t var = 0; var < NVARS; ++var) {
              u_node_n[var] = u(ielem, jelem, kelem, dof_n, var);
            }

            const std::array<T, 3> normal = {
                static_cast<T>(0.5) *
                    (contravariant_vectors(ielem, jelem, kelem, dof_i, 2, 0) +
                     contravariant_vectors(ielem, jelem, kelem, dof_n, 2, 0)),
                static_cast<T>(0.5) *
                    (contravariant_vectors(ielem, jelem, kelem, dof_i, 2, 1) +
                     contravariant_vectors(ielem, jelem, kelem, dof_n, 2, 1)),
                static_cast<T>(0.5) *
                    (contravariant_vectors(ielem, jelem, kelem, dof_i, 2, 2) +
                     contravariant_vectors(ielem, jelem, kelem, dof_n, 2, 2))};
            const auto flux_in =
                NumericFlux::numerical_flux(eq, u_node, u_node_n, normal);

            for (std::size_t var = 0; var < NVARS; ++var) {
              du(ielem, jelem, kelem, dof_i, var) +=
                  alpha * basis.derivative_split(knode, nnode) * flux_in[var];
              du(ielem, jelem, kelem, dof_n, var) +=
                  alpha * basis.derivative_split(nnode, knode) * flux_in[var];
            }
          }
        }
      }
    }
  }
};

template <class T, std::size_t NVARS, class NumericFlux, std::size_t NDIMS>
struct FinitVolumeIntegral;

template <class T, std::size_t NVARS, class NumericFlux>
struct FinitVolumeIntegral<T, NVARS, NumericFlux, 1> {
  template <class BasisData, class Equations, class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION static void
  fv_kernel(std::size_t ielem, const BasisData& basis, const Equations& eq,
            const ArrayU& u, ArrayDu& du, T alpha) {
    std::array<std::array<T, NVARS>, BasisData::NNodes + 1> fstar1_L{};
    std::array<std::array<T, NVARS>, BasisData::NNodes + 1> fstar1_R{};
    for (std::size_t inode = 1; inode < BasisData::NNodes; ++inode) {
      std::array<T, NVARS> u_ll{};
      std::array<T, NVARS> u_rr{};
      for (std::size_t var = 0; var < NVARS; ++var) {
        u_ll[var] = u(ielem, inode - 1, var);
        u_rr[var] = u(ielem, inode, var);
      }

      auto fstar1 = NumericFlux::numerical_flux(eq, u_ll, u_rr);
      for (std::size_t var = 0; var < NVARS; ++var) {
        fstar1_L[inode][var] = fstar1[var];
        fstar1_R[inode][var] = fstar1[var];
      }
    }

    for (std::size_t inode = 0; inode < BasisData::NNodes; ++inode) {
      for (std::size_t var = 0; var < NVARS; ++var) {
        du(ielem, inode, var) =
            du(ielem, inode, var) +
            (alpha * (basis.inv_weights[inode] *
                      (fstar1_L[inode + 1][var] - fstar1_R[inode][var])));
      }
    }
  }
};

template <class T, std::size_t NVARS, class NumericFlux>
struct FinitVolumeIntegral<T, NVARS, NumericFlux, 2> {
  template <class BasisData, class Equations, class ArrayU, class ArrayDu,
            class SubcellNormals>
  KOKKOS_INLINE_FUNCTION static void
  fv_kernel(std::size_t ielem, std::size_t jelem, const BasisData& basis,
            const SubcellNormals& subcell_normals, const Equations& eq,
            const ArrayU& u, ArrayDu& du, T alpha) {
    for (std::size_t jnode = 0; jnode < BasisData::NNodes; ++jnode) {
      for (std::size_t iface = 1; iface < BasisData::NNodes; ++iface) {
        const std::size_t left_dof =
            DGSEM::utils::local_dof<BasisData::NNodes>(iface - 1, jnode);
        const std::size_t right_dof =
            DGSEM::utils::local_dof<BasisData::NNodes>(iface, jnode);
        std::array<T, NVARS> u_ll{};
        std::array<T, NVARS> u_rr{};
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_ll[var] = u(ielem, jelem, left_dof, var);
          u_rr[var] = u(ielem, jelem, right_dof, var);
        }

        const auto fstar = NumericFlux::numerical_flux(
            eq, u_ll, u_rr,
            std::array<T, 2>{
                subcell_normals.normal_1(ielem, jelem, iface - 1, jnode, 0),
                subcell_normals.normal_1(ielem, jelem, iface - 1, jnode, 1)});
        for (std::size_t var = 0; var < NVARS; ++var) {
          du(ielem, jelem, left_dof, var) +=
              alpha * basis.inv_weights[iface - 1] * fstar[var];
          du(ielem, jelem, right_dof, var) -=
              alpha * basis.inv_weights[iface] * fstar[var];
        }
      }
    }

    for (std::size_t inode = 0; inode < BasisData::NNodes; ++inode) {
      for (std::size_t iface = 1; iface < BasisData::NNodes; ++iface) {
        const std::size_t bottom_dof =
            DGSEM::utils::local_dof<BasisData::NNodes>(inode, iface - 1);
        const std::size_t top_dof =
            DGSEM::utils::local_dof<BasisData::NNodes>(inode, iface);
        std::array<T, NVARS> u_ll{};
        std::array<T, NVARS> u_rr{};
        for (std::size_t var = 0; var < NVARS; ++var) {
          u_ll[var] = u(ielem, jelem, bottom_dof, var);
          u_rr[var] = u(ielem, jelem, top_dof, var);
        }

        const auto fstar = NumericFlux::numerical_flux(
            eq, u_ll, u_rr,
            std::array<T, 2>{
                subcell_normals.normal_2(ielem, jelem, inode, iface - 1, 0),
                subcell_normals.normal_2(ielem, jelem, inode, iface - 1, 1)});
        for (std::size_t var = 0; var < NVARS; ++var) {
          du(ielem, jelem, bottom_dof, var) +=
              alpha * basis.inv_weights[iface - 1] * fstar[var];
          du(ielem, jelem, top_dof, var) -=
              alpha * basis.inv_weights[iface] * fstar[var];
        }
      }
    }
  }
};

template <class T, std::size_t NVARS, class NumericFlux>
struct FinitVolumeIntegral<T, NVARS, NumericFlux, 3> {
  template <class BasisData, class Equations, class ArrayU, class ArrayDu,
            class SubcellNormals>
  KOKKOS_INLINE_FUNCTION static void
  fv_kernel(std::size_t ielem, std::size_t jelem, std::size_t kelem,
            const BasisData& basis, const SubcellNormals& subcell_normals,
            const Equations& eq, const ArrayU& u, ArrayDu& du, T alpha) {

    for (std::size_t knode = 0; knode < BasisData::NNodes; ++knode) {
      for (std::size_t jnode = 0; jnode < BasisData::NNodes; ++jnode) {
        for (std::size_t iface = 1; iface < BasisData::NNodes; ++iface) {
          const std::size_t left_dof =
              DGSEM::utils::local_dof<BasisData::NNodes>(iface - 1, jnode,
                                                         knode);
          const std::size_t right_dof =
              DGSEM::utils::local_dof<BasisData::NNodes>(iface, jnode, knode);

          std::array<T, NVARS> u_ll{};
          std::array<T, NVARS> u_rr{};
          for (std::size_t var = 0; var < NVARS; ++var) {
            u_ll[var] = u(ielem, jelem, kelem, left_dof, var);
            u_rr[var] = u(ielem, jelem, kelem, right_dof, var);
          }

          auto fstar = NumericFlux::numerical_flux(
              eq, u_ll, u_rr,
              std::array<T, 3>{
                  subcell_normals.normal_1(ielem, jelem, kelem, iface - 1,
                                           jnode, knode, 0),
                  subcell_normals.normal_1(ielem, jelem, kelem, iface - 1,
                                           jnode, knode, 1),
                  subcell_normals.normal_1(ielem, jelem, kelem, iface - 1,
                                           jnode, knode, 2)});

          for (std::size_t var = 0; var < NVARS; ++var) {
            du(ielem, jelem, kelem, left_dof, var) +=
                alpha * basis.inv_weights[iface - 1] * fstar[var];
            du(ielem, jelem, kelem, right_dof, var) -=
                alpha * basis.inv_weights[iface] * fstar[var];
          }
        }
      }
    }

    for (std::size_t knode = 0; knode < BasisData::NNodes; ++knode) {
      for (std::size_t jface = 1; jface < BasisData::NNodes; ++jface) {
        for (std::size_t inode = 0; inode < BasisData::NNodes; ++inode) {
          const std::size_t left_dof =
              DGSEM::utils::local_dof<BasisData::NNodes>(inode, jface - 1,
                                                         knode);
          const std::size_t right_dof =
              DGSEM::utils::local_dof<BasisData::NNodes>(inode, jface, knode);

          std::array<T, NVARS> u_ll{};
          std::array<T, NVARS> u_rr{};
          for (std::size_t var = 0; var < NVARS; ++var) {
            u_ll[var] = u(ielem, jelem, kelem, left_dof, var);
            u_rr[var] = u(ielem, jelem, kelem, right_dof, var);
          }

          auto fstar = NumericFlux::numerical_flux(
              eq, u_ll, u_rr,
              std::array<T, 3>{
                  subcell_normals.normal_2(ielem, jelem, kelem, inode,
                                           jface - 1, knode, 0),
                  subcell_normals.normal_2(ielem, jelem, kelem, inode,
                                           jface - 1, knode, 1),
                  subcell_normals.normal_2(ielem, jelem, kelem, inode,
                                           jface - 1, knode, 2)});

          for (std::size_t var = 0; var < NVARS; ++var) {
            du(ielem, jelem, kelem, left_dof, var) +=
                alpha * basis.inv_weights[jface - 1] * fstar[var];
            du(ielem, jelem, kelem, right_dof, var) -=
                alpha * basis.inv_weights[jface] * fstar[var];
          }
        }
      }
    }

    for (std::size_t kface = 1; kface < BasisData::NNodes; ++kface) {
      for (std::size_t jnode = 0; jnode < BasisData::NNodes; ++jnode) {
        for (std::size_t inode = 0; inode < BasisData::NNodes; ++inode) {
          const std::size_t left_dof =
              DGSEM::utils::local_dof<BasisData::NNodes>(inode, jnode,
                                                         kface - 1);
          const std::size_t right_dof =
              DGSEM::utils::local_dof<BasisData::NNodes>(inode, jnode, kface);

          std::array<T, NVARS> u_ll{};
          std::array<T, NVARS> u_rr{};
          for (std::size_t var = 0; var < NVARS; ++var) {
            u_ll[var] = u(ielem, jelem, kelem, left_dof, var);
            u_rr[var] = u(ielem, jelem, kelem, right_dof, var);
          }

          auto fstar = NumericFlux::numerical_flux(
              eq, u_ll, u_rr,
              std::array<T, 3>{
                  subcell_normals.normal_3(ielem, jelem, kelem, inode, jnode,
                                           kface - 1, 0),
                  subcell_normals.normal_3(ielem, jelem, kelem, inode, jnode,
                                           kface - 1, 1),
                  subcell_normals.normal_3(ielem, jelem, kelem, inode, jnode,
                                           kface - 1, 2)});

          for (std::size_t var = 0; var < NVARS; ++var) {
            du(ielem, jelem, kelem, left_dof, var) +=
                alpha * basis.inv_weights[kface - 1] * fstar[var];
            du(ielem, jelem, kelem, right_dof, var) -=
                alpha * basis.inv_weights[kface] * fstar[var];
          }
        }
      }
    }
  }
};

} // namespace detail

template <class Basis, class Equations>
struct VolumeIntegralWeakForm {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;
  using BasisData = typename Basis::DeviceData;
  using MetricArray =
      typename jacobian_type_traits<value_type, traits::NDIMS>::JacobianMatrix;

  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  VolumeIntegralWeakForm() : basis_data(Basis::device_data()) {}
  explicit VolumeIntegralWeakForm(MetricArray contravariant_vectors_)
      : basis_data(Basis::device_data()),
        contravariant_vectors(contravariant_vectors_) {}

  template <class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t ielem, const Equations& eq,
                                         const ArrayU& u, ArrayDu& du) const
    requires(NDIMS == 1)
  {
    detail::VolumeIntegralWeak<value_type, NVARS, NDIMS>::
        template weak_form_kernel<BasisData, Equations>(ielem, basis_data, eq,
                                                        u, du);
  }

  template <class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t ielem, std::size_t jelem,
                                         const Equations& eq, const ArrayU& u,
                                         ArrayDu& du) const
    requires(NDIMS == 2)
  {
    detail::VolumeIntegralWeak<value_type, NVARS, NDIMS>::
        template weak_form_kernel<BasisData, Equations>(
            ielem, jelem, basis_data, contravariant_vectors, eq, u, du);
  }

  template <class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t ielem, std::size_t jelem,
                                         std::size_t kelem, const Equations& eq,
                                         const ArrayU& u, ArrayDu& du) const
    requires(NDIMS == 3)
  {
    detail::VolumeIntegralWeak<value_type, NVARS, NDIMS>::
        template weak_form_kernel<BasisData, Equations>(
            ielem, jelem, kelem, basis_data, contravariant_vectors, eq, u, du);
  }

  BasisData basis_data;
  MetricArray contravariant_vectors;
};

template <class Basis, class Equations, template <class> class NumericFlux>
struct VolumeIntegralSplitForm {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;
  using BasisData = typename Basis::DeviceData;
  using MetricArray =
      typename jacobian_type_traits<value_type, traits::NDIMS>::JacobianMatrix;

  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  VolumeIntegralSplitForm() : basis_data(Basis::device_data()) {}
  explicit VolumeIntegralSplitForm(MetricArray contravariant_vectors_)
      : basis_data(Basis::device_data()),
        contravariant_vectors(contravariant_vectors_) {}

  template <class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t ielem, const Equations& eq,
                                         const ArrayU& u, ArrayDu& du) const
    requires(NDIMS == 1)
  {
    detail::VolumeIntegralSplit<value_type, NVARS, NumericFlux<Equations>,
                                NDIMS>::template split_form_kernel<BasisData,
                                                                   Equations>(
        ielem, basis_data, eq, u, du);
  }

  template <class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t ielem, std::size_t jelem,
                                         const Equations& eq, const ArrayU& u,
                                         ArrayDu& du) const
    requires(NDIMS == 2)
  {
    detail::VolumeIntegralSplit<value_type, NVARS, NumericFlux<Equations>,
                                NDIMS>::template split_form_kernel<BasisData,
                                                                   Equations>(
        ielem, jelem, basis_data, contravariant_vectors, eq, u, du);
  }

  template <class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t ielem, std::size_t jelem,
                                         std::size_t kelem, const Equations& eq,
                                         const ArrayU& u, ArrayDu& du) const
    requires(NDIMS == 3)
  {
    detail::VolumeIntegralSplit<value_type, NVARS, NumericFlux<Equations>,
                                NDIMS>::template split_form_kernel<BasisData,
                                                                   Equations>(
        ielem, jelem, kelem, basis_data, contravariant_vectors, eq, u, du);
  }

  BasisData basis_data;
  MetricArray contravariant_vectors;
};

struct VolumeIntegralShockCapturingBase {};

template <class Basis, class Equations, template <class> class NumericFlux,
          template <class> class FvFlux, class Indicator>
struct VolumeIntegralShockCapturingHG
    : public VolumeIntegralShockCapturingBase {
  using traits = equations::EquationTraits<Equations>;
  using value_type = typename traits::value_type;
  using BasisData = typename Basis::DeviceData;

  constexpr static std::size_t NDIMS = traits::NDIMS;
  constexpr static std::size_t NVARS = traits::NVARS;

  using AlphaView = solution_type_traits<value_type, NDIMS>::ElementArray;
  using DataArray = solution_type_traits<value_type, NDIMS>::DataArray;
  using MetricArray =
      typename jacobian_type_traits<value_type, NDIMS>::JacobianMatrix;
  using SubcellNormals =
      typename SubcellNormalVectors<value_type, NDIMS>::DeviceData;

  VolumeIntegralShockCapturingHG() = delete;

  VolumeIntegralShockCapturingHG(
      value_type alpha_max_, value_type alpha_min_, bool alpha_smooth_,
      const std::array<std::size_t, NDIMS>& n_cells_,
      MetricArray contravariant_vectors_ = MetricArray(),
      SubcellNormals subcell_normals_ = SubcellNormals())
      : indicator(alpha_max_, alpha_min_, alpha_smooth_, Basis::device_data()),
        basis_data(Basis::device_data()),
        contravariant_vectors(contravariant_vectors_),
        subcell_normals(subcell_normals_) {
    value_type eps_val = std::numeric_limits<value_type>::epsilon();
    atol = std::max(100.0 * eps_val, std::pow(eps_val, 0.75));

    if constexpr (AlphaView::rank == 1) {
      alpha_view = AlphaView("alpha_view", n_cells_[0]);
    } else if constexpr (AlphaView::rank == 2) {
      alpha_view = AlphaView("alpha_view", n_cells_[0], n_cells_[1]);
    } else if constexpr (AlphaView::rank == 3) {
      alpha_view =
          AlphaView("alpha_view", n_cells_[0], n_cells_[1], n_cells_[2]);
    }
  }

  inline void calc_alpha(const std::array<std::size_t, NDIMS>& n_cells,
                         DataArray u) {
    indicator(n_cells, u, alpha_view);
  }

  template <class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t ielem, const Equations& eq,
                                         const ArrayU& u, ArrayDu& du) const
    requires(NDIMS == 1)
  {
    value_type alpha_ielem = alpha_view(ielem);
    bool dg_only = std::abs(alpha_ielem - value_type{0.0}) <= atol;

    if (dg_only) {
      detail::VolumeIntegralSplit<value_type, NVARS, NumericFlux<Equations>,
                                  NDIMS>::template split_form_kernel<BasisData,
                                                                     Equations>(
          ielem, basis_data, eq, u, du);
    } else {
      detail::VolumeIntegralSplit<value_type, NVARS, NumericFlux<Equations>,
                                  NDIMS>::template split_form_kernel<BasisData,
                                                                     Equations>(
          ielem, basis_data, eq, u, du,
          (static_cast<value_type>(1.0) - alpha_ielem));

      detail::FinitVolumeIntegral<value_type, NVARS, FvFlux<Equations>, NDIMS>::
          template fv_kernel<BasisData, Equations>(ielem, basis_data, eq, u, du,
                                                   alpha_ielem);
    }
  }

  template <class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t ielem, std::size_t jelem,
                                         const Equations& eq, const ArrayU& u,
                                         ArrayDu& du) const
    requires(NDIMS == 2)
  {
    const value_type alpha_ielem = alpha_view(ielem, jelem);
    const bool dg_only = std::abs(alpha_ielem - value_type{0.0}) <= atol;

    if (dg_only) {
      detail::VolumeIntegralSplit<value_type, NVARS, NumericFlux<Equations>,
                                  NDIMS>::template split_form_kernel<BasisData,
                                                                     Equations>(
          ielem, jelem, basis_data, contravariant_vectors, eq, u, du);
    } else {
      detail::VolumeIntegralSplit<value_type, NVARS, NumericFlux<Equations>,
                                  NDIMS>::template split_form_kernel<BasisData,
                                                                     Equations>(
          ielem, jelem, basis_data, contravariant_vectors, eq, u, du,
          (static_cast<value_type>(1.0) - alpha_ielem));

      detail::FinitVolumeIntegral<value_type, NVARS, FvFlux<Equations>,
                                  NDIMS>::template fv_kernel<BasisData,
                                                             Equations>(
          ielem, jelem, basis_data, subcell_normals, eq, u, du, alpha_ielem);
    }
  }

  template <class ArrayU, class ArrayDu>
  KOKKOS_INLINE_FUNCTION void operator()(std::size_t ielem, std::size_t jelem,
                                         std::size_t kelem, const Equations& eq,
                                         const ArrayU& u, ArrayDu& du) const
    requires(NDIMS == 3)
  {
    const value_type alpha_ielem = alpha_view(ielem, jelem, kelem);
    const bool dg_only = std::abs(alpha_ielem - value_type{0.0}) <= atol;

    if (dg_only) {
      detail::VolumeIntegralSplit<value_type, NVARS, NumericFlux<Equations>,
                                  NDIMS>::template split_form_kernel<BasisData,
                                                                     Equations>(
          ielem, jelem, kelem, basis_data, contravariant_vectors, eq, u, du);
    } else {
      detail::VolumeIntegralSplit<value_type, NVARS, NumericFlux<Equations>,
                                  NDIMS>::template split_form_kernel<BasisData,
                                                                     Equations>(
          ielem, jelem, kelem, basis_data, contravariant_vectors, eq, u, du,
          (static_cast<value_type>(1.0) - alpha_ielem));

      detail::FinitVolumeIntegral<value_type, NVARS, FvFlux<Equations>, NDIMS>::
          template fv_kernel<BasisData, Equations>(ielem, jelem, kelem,
                                                   basis_data, subcell_normals,
                                                   eq, u, du, alpha_ielem);
    }
  }

private:
  Indicator indicator;
  AlphaView alpha_view;
  BasisData basis_data;
  MetricArray contravariant_vectors;
  SubcellNormals subcell_normals;
  value_type atol;
};
} // namespace DGSEM
