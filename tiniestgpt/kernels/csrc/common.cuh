// ---------------------------------------------------------------------------
// 公共头文件：所有 kernel 共用的宏与工具。
//
// 注意：本文件**不是**通过 #include 引入的，而是由 ``loader.py`` 在 JIT 编译时
// 以文本方式拼接到每个 .cu 前面。这样做的原因是 torch 的 ``load_inline``
// 会把每份 source 写到临时目录并用不同文件名编译，include 路径不可控；
// 直接拼接最稳。
// ---------------------------------------------------------------------------
#pragma once

// TG_STANDALONE：不依赖 PyTorch 单独编译（scripts/verify_cuda_kernels.py 用）。
// 这样在没有"torch 头文件与 nvcc 版本互相兼容"的环境里，也能验证内核本身的正确性。
#ifdef TG_STANDALONE
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#define TG_CUDA_CHECK(call)                                                   \
  do {                                                                        \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error at %s:%d -> %s\n", __FILE__, __LINE__,      \
              cudaGetErrorString(err__));                                     \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)
#else
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <cfloat>
#include <cstdio>
#define TG_CUDA_CHECK(call)                                                   \
  do {                                                                        \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      TORCH_CHECK(false, "CUDA error at ", __FILE__, ":", __LINE__, " -> ",   \
                  cudaGetErrorString(err__));                                 \
    }                                                                         \
  } while (0)
// 所有 kernel 统一走当前 PyTorch 流，保证与 autograd / CUDA Graph 的时序正确。
#define TG_STREAM at::cuda::getCurrentCUDAStream()
#endif

#define TG_CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#define TG_SOFTMAX_BLOCK 256
#define TG_TRANSPOSE_TILE 32

// 注意：必须同时标注 __host__ __device__ —— 这个函数既在 host 端算 grid，
// 也会在 kernel 内部被调用（少了 __device__ 会报
// "calling a __host__ function from a __global__ function is not allowed"）。
__host__ __device__ __forceinline__ int tg_ceil_div_i(int a, int b) { return (a + b - 1) / b; }
