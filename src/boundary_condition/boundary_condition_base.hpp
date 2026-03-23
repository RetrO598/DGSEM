#pragma once

namespace DGSEM {
enum class BoundaryCondition {
  Periodic,
  Wall,
  Inflow,
  Outflow,
  Extrapolate,
  Dirichlet,
  Custom,
};
}