// ---------------------------------------------------------------------------
// 01 · 第一个实用 kernel：向量加法
//
// 教学要点（路线 1.4）：
//   * 线程索引 ``i = blockIdx.x * blockDim.x + threadIdx.x`` 是 CUDA 的"身份证"；
//   * ``if (i < n)`` 的边界检查是必须的——grid 通常是向上取整的，尾部会多出线程；
//   * ``__restrict__`` 告诉编译器三个指针不重叠，允许更激进的访存优化；
//   * kernel 是**异步**的：launch 后立即返回，错误要等到下一次同步才暴露。
// ---------------------------------------------------------------------------

__global__ void vector_add_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

#ifndef TG_STANDALONE
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.device().is_cuda() && b.device().is_cuda(), "vector_add 需要 CUDA 张量");
  TORCH_CHECK(a.numel() == b.numel(), "vector_add: 形状不一致");
  const int n = static_cast<int>(a.numel());
  auto out = torch::empty_like(a);
  if (n == 0) return out;

  const int block = 256;
  const int grid = tg_ceil_div_i(n, block);
  vector_add_kernel<<<grid, block, 0, TG_STREAM>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
  return out;
}
#endif  // TG_STANDALONE
