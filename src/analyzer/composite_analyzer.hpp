#pragma once

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <equations/equations.hpp>
#include <tuple>
#include <utility>
namespace DGSEM {
template <class Basis, class Equations, class... Analyzers>
struct CompositeAnalyzer {
  std::tuple<Analyzers...> analyzers;

  using trait = equations::EquationTraits<Equations>;
  using T = trait::value_type;

  constexpr static std::size_t NDIMS = trait::NDIMS;
  constexpr static std::size_t NVARS = trait::NVARS;

  using index_seq = std::index_sequence_for<Analyzers...>;

  KOKKOS_INLINE_FUNCTION
  void init() { init_impl(index_seq{}); }

  template <std::size_t... I>
  KOKKOS_INLINE_FUNCTION void init_impl(std::index_sequence<I...>) {
    ((std::get<I>(analyzers).init()), ...);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const Kokkos::Array<T, NVARS>& u) {
    apply_impl(u, index_seq{});
  }

  template <std::size_t... I>
  KOKKOS_INLINE_FUNCTION void apply_impl(const Kokkos::Array<T, NVARS>& u,
                                         std::index_sequence<I...>) {
    ((std::get<I>(analyzers)(u)), ...);
  }

  KOKKOS_INLINE_FUNCTION
  void join(const CompositeAnalyzer& other) { join_impl(other, index_seq{}); }

  template <std::size_t... I>
  KOKKOS_INLINE_FUNCTION void join_impl(const CompositeAnalyzer& other,
                                        std::index_sequence<I...>) {
    ((std::get<I>(analyzers).join(std::get<I>(other.analyzers))), ...);
  }

  template <typename Type>
  KOKKOS_INLINE_FUNCTION Type& get() {
    return get_impl<Type, 0>();
  }

  template <typename Type, std::size_t I>
  KOKKOS_INLINE_FUNCTION Type& get_impl() {
    if constexpr (I >= sizeof...(Analyzers)) {
      static_assert(I < sizeof...(Analyzers), "Type not found");
    } else if constexpr (std::is_same_v<Type,
                                        std::tuple_element_t<
                                            I, std::tuple<Analyzers...>>>) {
      return std::get<I>(analyzers);
    } else {
      return get_impl<Type, I + 1>();
    }
  }
};
} // namespace DGSEM