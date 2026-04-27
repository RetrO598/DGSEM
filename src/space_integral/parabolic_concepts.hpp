#pragma once

#include <array>
#include <cstddef>
#include <equations/equations.hpp>

namespace DGSEM {

template <class Equations>
concept ParabolicEquations = requires(
    const Equations& eq,
    const std::array<typename equations::EquationTraits<Equations>::value_type,
                     equations::EquationTraits<Equations>::NVARS>& u,
    const std::array<
        std::array<typename equations::EquationTraits<Equations>::value_type,
                   Equations::NGRAD_VARS>,
        equations::EquationTraits<Equations>::NDIMS>& grad_q) {
  Equations::NGRAD_VARS;
  { eq.gradient_variables(u) };
  { eq.viscous_flux(eq.gradient_variables(u), grad_q, std::size_t{0}) };
};

} // namespace DGSEM
