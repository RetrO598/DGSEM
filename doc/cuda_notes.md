# CUDA and Kokkos Notes

DGSEM uses Kokkos to target CPU and GPU backends. CUDA builds are more sensitive
to host/device annotations and template complexity.

## Device Functions

Any function called inside a Kokkos kernel must be marked:

```cpp
KOKKOS_INLINE_FUNCTION
```

Avoid host-only APIs in these functions.

## Basis Lifecycle

Basis data is stored in static Kokkos views. The safe lifecycle is:

```cpp
Kokkos::initialize();
{
  using Basis = DGSEM::Basis::LobattoLegendreBasis<double, 4>;
  Basis::initialize();
  // simulation
  Basis::finalize();
}
Kokkos::finalize();
```

Examples may use `DGSEM::KokkosSession` and `DGSEM::BasisGuard<Basis>` to manage
this lifecycle.

## NVCC Sensitivity

NVCC can be sensitive to complex C++20 template constraints in device-heavy
code. Prefer simple traits, `if constexpr`, and explicit static checks when
possible.

## Data Movement

Large host-device transfers are usually visible as `Kokkos::deep_copy`.
Geometry initialization calls `sync_to_device()` once after host-side geometry
construction. Time stepping mostly works on device views.

