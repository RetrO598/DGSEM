# AGENTS.md

## 项目概述
DGSEM 是一个高性能的计算流体力学（CFD）求解器，基于高阶谱元法（DGSEM）。代码库采用模块化设计，主要以头文件为主，支持 GPU 加速。

## 架构概述
代码库的主要模块包括：
- `src/dgsem.hpp`：主入口，包含所有库组件。
- `src/base/`：核心定义、特性和数学工具。
- `src/mesh/`：网格定义，主要支持 `StructuredMesh`。
- `src/equations/`：物理模型（如 `CompressibleEuler1D`）。
- `src/solver/`：核心求解器，协调 DGSEM 算法。
- `src/time_integrator/`：时间离散化方案（如 `SSPRK3`）。

更多详细信息，请参考 [GEMINI.md](GEMINI.md)。

## 构建和运行

### 依赖
- **C++20 编译器**：GCC 11+ 或 Clang 15+。
- **CUDA（可选）**：支持 GPU 的 CUDA 12.1 或更高版本。
- **CMake**：3.28+。
- **Kokkos**：由 CMake 自动 Fetch Kokkos 4.7.02，默认构建不需要单独安装。

### 构建说明
#### CPU/OpenMP
```bash
mkdir build && cd build
cmake .. -DENABLE_CUDA=OFF -DENABLE_OPENMP=ON -DENABLE_SERIAL=ON
make -j$(nproc)
```

#### GPU/CUDA
```bash
mkdir build && cd build
cmake .. -DENABLE_CUDA=ON -DENABLE_OPENMP=OFF -DENABLE_SERIAL=ON
make -j$(nproc)
```

更多构建选项，请参考 [GEMINI.md](GEMINI.md)。

## 开发约定
- **多项式基函数生命周期**：优先使用 `DGSEM::BasisGuard<Basis>` 管理初始化和释放。
- **Kokkos 生命周期**：应用和示例代码可使用 `DGSEM::KokkosSession` 管理 `Kokkos::initialize()` / `Kokkos::finalize()`。
- **结构网格装配**：优先使用 `DGSEM::make_structured_problem` 装配 mesh、element cache、solver 和 solution；底层 `StructuredSolver` API 仍可直接使用。
- **GPU 性能优化**：优先使用 Kokkos 提供的并行化工具。
- **头文件结构**：尽量保持模块化，避免循环依赖。

更多开发约定，请参考 [GEMINI.md](GEMINI.md)。

## 常见问题
1. **构建失败**：
   - 确保安装了正确版本的依赖。
   - 检查 CMake 输出中的错误信息。
2. **运行时错误**：
   - 确保输入参数正确。
   - 使用调试模式运行以捕获更多信息。

## 测试
测试代码位于 `examples/` 目录下。可以通过以下命令运行：
```bash
ctest --output-on-failure
```
