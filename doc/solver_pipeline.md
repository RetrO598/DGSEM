# Solver Pipeline

The central spatial routine is:

```cpp
StructuredSolver::calc_rhs(solution& sol, value_type time)
```

## Hyperbolic Terms

The inviscid part follows this order:

```text
clear RHS
volume integral
interface flux
boundary flux
surface integral
Jacobian projection
```

## Parabolic Terms

For equations satisfying `ParabolicEquations`, additional work is performed:

```text
clear gradient
gradient volume integral
transform gradient to physical coordinates
gradient interface flux
gradient boundary condition
gradient surface integral
gradient Jacobian projection
viscous flux
viscous volume integral
viscous interface flux
viscous boundary condition
viscous surface integral
```

## Important Files

- `src/solver/structured_solver.hpp`
- `src/space_integral/volume_integral_functor.hpp`
- `src/space_integral/surface_flux_functor.hpp`
- `src/space_integral/surface_integral_functor.hpp`
- `src/space_integral/parabolic_gradient.hpp`
- `src/space_integral/parabolic_viscous.hpp`

## Notes

The pipeline intentionally keeps algorithmic orchestration in the solver and
kernel-level work in functors. This helps preserve Kokkos portability while
keeping the solver flow readable.

