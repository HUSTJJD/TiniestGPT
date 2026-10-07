// ---------------------------------------------------------------------------
// 05 · Bank Conflict 直觉实验：32×32 矩阵转置
//
// 共享内存被分成 32 个 bank（每 bank 4 字节）。一个 warp 的 32 个线程
// **同时**访问同一个 bank 的不同地址时，这些访问会被串行化 → bank conflict。
//
// 转置是教科书级的反例：
//   * 读阶段 tile[ty][tx]，线程按 tx 变化 → 连续地址 → 无冲突
//   * 写阶段 tile[tx][ty]，线程按 tx 变化但行宽正好 32 → 全部落到同一 bank
//     → 32-way bank conflict，带宽直接掉到 1/32
//
// 修法只需要一行：**给共享数组加一列 padding**（32 → 33），
// 让每行错开一个 bank，冲突立刻消失。这就是"加一列就快 30 倍"的由来。
// ---------------------------------------------------------------------------

// ------------------------- 有冲突版本（朴素） ----------------------------- //
__global__ void transpose_naive_kernel(const float* __restrict__ in,
                                       float* __restrict__ out, int W, int H) {
  __shared__ float tile[TG_TRANSPOSE_TILE][TG_TRANSPOSE_TILE];

  int x = blockIdx.x * TG_TRANSPOSE_TILE + threadIdx.x;
  int y = blockIdx.y * TG_TRANSPOSE_TILE + threadIdx.y;
  if (x < W && y < H) {
    tile[threadIdx.y][threadIdx.x] = in[(size_t)y * W + x];
  }
  __syncthreads();

  x = blockIdx.y * TG_TRANSPOSE_TILE + threadIdx.x;
  y = blockIdx.x * TG_TRANSPOSE_TILE + threadIdx.y;
  if (x < H && y < W) {
    out[(size_t)y * H + x] = tile[threadIdx.x][threadIdx.y];   // ← 32-way conflict
  }
}

// ----------------------- 无冲突版本（+1 列 padding） ---------------------- //
__global__ void transpose_padded_kernel(const float* __restrict__ in,
                                        float* __restrict__ out, int W, int H) {
  __shared__ float tile[TG_TRANSPOSE_TILE][TG_TRANSPOSE_TILE + 1];

  int x = blockIdx.x * TG_TRANSPOSE_TILE + threadIdx.x;
  int y = blockIdx.y * TG_TRANSPOSE_TILE + threadIdx.y;
  if (x < W && y < H) {
    tile[threadIdx.y][threadIdx.x] = in[(size_t)y * W + x];
  }
  __syncthreads();

  x = blockIdx.y * TG_TRANSPOSE_TILE + threadIdx.x;
  y = blockIdx.x * TG_TRANSPOSE_TILE + threadIdx.y;
  if (x < H && y < W) {
    out[(size_t)y * H + x] = tile[threadIdx.x][threadIdx.y];   // ← 每行错开 1 bank
  }
}

// ---------------------------------------------------------------------------
#ifndef TG_STANDALONE
torch::Tensor transpose(torch::Tensor x, int64_t padded) {
  TORCH_CHECK(x.device().is_cuda(), "transpose 需要 CUDA 张量");
  TORCH_CHECK(x.dim() == 2, "transpose 只支持 2D [H, W]");

  const int H = static_cast<int>(x.size(0));
  const int W = static_cast<int>(x.size(1));
  auto out = torch::empty({W, H}, x.options());
  if (H == 0 || W == 0) return out;

  dim3 block(TG_TRANSPOSE_TILE, TG_TRANSPOSE_TILE);
  dim3 grid(tg_ceil_div_i(W, TG_TRANSPOSE_TILE), tg_ceil_div_i(H, TG_TRANSPOSE_TILE));
  if (padded) {
    transpose_padded_kernel<<<grid, block, 0, TG_STREAM>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), W, H);
  } else {
    transpose_naive_kernel<<<grid, block, 0, TG_STREAM>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), W, H);
  }
  return out;
}
#endif  // TG_STANDALONE
