# Extending Equations

An equation type supplies physical fluxes and compile-time dimensions.

## Minimal Shape

```cpp
template <class T>
class MyEquation {
public:
  using value_type = T;

  static constexpr std::size_t NDIMS = 1;
  static constexpr std::size_t NVARS = 1;

  KOKKOS_INLINE_FUNCTION
  std::array<T, NVARS> flux(const std::array<T, NVARS>& u,
                            std::size_t dim) const {
    return {/* ... */};
  }

  KOKKOS_INLINE_FUNCTION
  T get_wave_speed(const std::array<T, NVARS>& ul,
                   const std::array<T, NVARS>& ur) const {
    return T{};
  }
};
```

## Parabolic Equations

Parabolic equations provide extra gradient and viscous-flux APIs. See the
Navier-Stokes equation headers for the current expected interface.

## Device Compatibility

Functions called inside Kokkos kernels must be marked with
`KOKKOS_INLINE_FUNCTION` and must avoid host-only facilities.

## Checklist

- Define `value_type`, `NDIMS`, and `NVARS`.
- Implement inviscid flux behavior.
- Implement wave-speed estimates used by numerical fluxes.
- Add primitive/conservative conversion helpers if needed.
- Add a small example before relying on the new equation in production cases.

