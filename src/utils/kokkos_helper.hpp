#pragma once

#include <base/base.hpp>
#include <iostream>

namespace DGSEM {
template <class ViewType>
void clone_view_shape(ViewType &dst, const ViewType &src) {
  if constexpr (ViewType::rank == 1) {
    Kokkos::realloc(dst, src.extent(0));
  } else if constexpr (ViewType::rank == 2) {
    Kokkos::realloc(dst, src.extent(0), src.extent(1));
  } else if constexpr (ViewType::rank == 3) {
    Kokkos::realloc(dst, src.extent(0), src.extent(1), src.extent(2));
  } else if constexpr (ViewType::rank == 4) {
    Kokkos::realloc(dst, src.extent(0), src.extent(1), src.extent(2),
                    src.extent(3));
  } else {
    static_assert(ViewType::rank <= 4, "Unsupported rank");
  }

  Kokkos::deep_copy(dst, 0.0);
}

template <class ViewType, class XArray>
void copy_xtensor_to_kokkos(ViewType &view, const XArray &arr) {
  constexpr int rank = ViewType::rank;

  // -------- 分配 View --------
  if constexpr (rank == 1) {
    Kokkos::realloc(view, arr.shape()[0]);
  } else if constexpr (rank == 2) {
    Kokkos::realloc(view, arr.shape()[0], arr.shape()[1]);
  } else if constexpr (rank == 3) {
    Kokkos::realloc(view, arr.shape()[0], arr.shape()[1], arr.shape()[2]);
  } else if constexpr (rank == 4) {
    Kokkos::realloc(view, arr.shape()[0], arr.shape()[1], arr.shape()[2],
                    arr.shape()[3]);
  }

  // -------- 创建 mirror --------
  auto host = Kokkos::create_mirror_view(view);

  // -------- 拷贝数据 --------
  auto shape = arr.shape();

  if constexpr (rank == 3) {
    for (size_t i = 0; i < shape[0]; ++i)
      for (size_t j = 0; j < shape[1]; ++j)
        for (size_t k = 0; k < shape[2]; ++k)
          host(i, j, k) = arr(i, j, k);
  } else if constexpr (rank == 4) {
    for (size_t i = 0; i < shape[0]; ++i)
      for (size_t j = 0; j < shape[1]; ++j)
        for (size_t k = 0; k < shape[2]; ++k)
          for (size_t l = 0; l < shape[3]; ++l)
            host(i, j, k, l) = arr(i, j, k, l);
  } else if constexpr (rank == 2) {
    for (size_t i = 0; i < shape[0]; ++i)
      for (size_t j = 0; j < shape[1]; ++j)
        host(i, j) = arr(i, j);
  }

  // -------- deep_copy 到 device --------
  Kokkos::deep_copy(view, host);
}

template <class ViewType, class XArray>
double compare_view_xtensor(const ViewType &view, const XArray &arr,
                            double tol = 1e-12, bool verbose = true) {
  using value_type = typename ViewType::non_const_value_type;

  constexpr int rank = ViewType::rank;

  // -------- 1. rank 检查 --------
  if (rank != static_cast<int>(arr.dimension())) {
    std::cerr << "Rank mismatch\n";
    return -1.0;
  }

  // -------- 2. shape 检查 --------
  for (int d = 0; d < rank; ++d) {
    if (view.extent(d) != arr.shape()[d]) {
      std::cerr << "Shape mismatch at dim " << d << "\n";
      return -1.0;
    }
  }

  // -------- 3. mirror --------
  auto host = Kokkos::create_mirror_view(view);
  Kokkos::deep_copy(host, view);

  double max_error = 0.0;
  size_t mismatch_count = 0;

  // ===============================
  // 元素比较函数（关键改进）
  // ===============================
  auto check = [&](auto a, auto b, auto... idx) {
    if constexpr (std::is_floating_point_v<value_type>) {
      double err = std::abs(a - b);
      if (err > tol) {
        if (verbose) {
          std::cout << "Mismatch (";
          ((std::cout << idx << ","), ...);
          std::cout << "): " << a << " vs " << b << " err=" << err << "\n";
        }
        mismatch_count++;
      }
      max_error = std::max(max_error, err);
    } else {
      // 整型：直接比较
      if (a != b) {
        if (verbose) {
          std::cout << "Mismatch (";
          ((std::cout << idx << ","), ...);
          std::cout << "): " << a << " vs " << b << "\n";
        }
        mismatch_count++;
      }
    }
  };

  // ===============================
  // 展开循环
  // ===============================
  if constexpr (rank == 1) {
    for (size_t i = 0; i < arr.shape()[0]; ++i)
      check(host(i), arr(i), i);
  }

  else if constexpr (rank == 2) {
    for (size_t i = 0; i < arr.shape()[0]; ++i)
      for (size_t j = 0; j < arr.shape()[1]; ++j)
        check(host(i, j), arr(i, j), i, j);
  }

  else if constexpr (rank == 3) {
    for (size_t i = 0; i < arr.shape()[0]; ++i)
      for (size_t j = 0; j < arr.shape()[1]; ++j)
        for (size_t k = 0; k < arr.shape()[2]; ++k)
          check(host(i, j, k), arr(i, j, k), i, j, k);
  }

  else if constexpr (rank == 4) {
    for (size_t i = 0; i < arr.shape()[0]; ++i)
      for (size_t j = 0; j < arr.shape()[1]; ++j)
        for (size_t k = 0; k < arr.shape()[2]; ++k)
          for (size_t l = 0; l < arr.shape()[3]; ++l)
            check(host(i, j, k, l), arr(i, j, k, l), i, j, k, l);
  }

  else {
    std::cerr << "Unsupported rank\n";
    return -1.0;
  }

  // -------- summary --------
  if (verbose) {
    std::cout << "Mismatch count = " << mismatch_count << "\n";
    if constexpr (std::is_floating_point_v<value_type>) {
      std::cout << "Max error = " << max_error << "\n";
    }
  }

  return max_error;
}
} // namespace DGSEM