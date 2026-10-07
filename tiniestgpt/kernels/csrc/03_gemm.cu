// ---------------------------------------------------------------------------
// 03 · GEMM：从朴素到共享内存分块（路线 1.3 的"GEMM 分块"检验标准）
//
// 为什么朴素版慢：C 的每个元素都要从 HBM 读一整行 A + 一整列 B，
// 算术强度 = 2K flops / 2K loads ≈ 1 → 必然被显存带宽卡死。
//
// 分块（tiling）的思路：把 A、B 切成 TILE×TILE 的小块搬进**共享内存**，
// 一个 block 负责算 C 的一个 TILE×TILE 子块，于是每块数据被复用 TILE 次，
// HBM 访存量从 O(M·N·K) 降到 O(M·N·K/TILE)。
//
// 进一步优化方向（本项目留给读者）：
//   * 每线程算 4×4 子块（提高寄存器复用）
//   * 用 float4 做 128-bit 向量化访存
//   * 双缓冲 / async copy（Ampere+）
//   * 换 Tensor Core（wmma / mma 指令）→ 这才是 cuBLAS 快的根本原因
// ---------------------------------------------------------------------------

// --------------------------- 朴素版（对照用） ----------------------------- //
__global__ void gemm_naive_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C, int M, int N, int K) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    acc += A[row * K + k] * B[k * N + col];   // B 是按列读的 → 非合并访问
  }
  C[row * N + col] = acc;
}

// --------------------------- 共享内存分块版 ------------------------------- //
template <int TILE>
__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;      // C 的全局行
  const int col = blockIdx.x * TILE + tx;      // C 的全局列

  float acc = 0.f;
  const int n_tiles = tg_ceil_div_i(K, TILE);
  for (int t = 0; t < n_tiles; ++t) {
    const int a_k = t * TILE + tx;             // A[row][a_k]
    const int b_k = t * TILE + ty;             // B[b_k][col]

    // 越界补 0：这样内层循环不用再判断边界
    As[ty][tx] = (row < M && a_k < K) ? A[row * K + a_k] : 0.f;
    Bs[ty][tx] = (b_k < K && col < N) ? B[b_k * N + col] : 0.f;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE; ++k) {
      acc += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();                           // 下一轮写入前必须等所有人算完
  }
  if (row < M && col < N) {
    C[row * N + col] = acc;
  }
}

// ---------------------------------------------------------------------------
#ifndef TG_STANDALONE
torch::Tensor gemm(torch::Tensor a, torch::Tensor b, int64_t tiled) {
  TORCH_CHECK(a.device().is_cuda() && b.device().is_cuda(), "gemm 需要 CUDA 张量");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "gemm 只支持 2D 输入");
  TORCH_CHECK(a.size(1) == b.size(0), "gemm: 维度不匹配");

  const int M = static_cast<int>(a.size(0));
  const int K = static_cast<int>(a.size(1));
  const int N = static_cast<int>(b.size(1));
  auto out = torch::empty({M, N}, a.options());
  if (M == 0 || N == 0 || K == 0) return out;

  const float* ap = a.data_ptr<float>();
  const float* bp = b.data_ptr<float>();
  float* cp = out.data_ptr<float>();

  if (tiled) {
    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid(tg_ceil_div_i(N, TILE), tg_ceil_div_i(M, TILE));
    gemm_tiled_kernel<TILE><<<grid, block, 0, TG_STREAM>>>(ap, bp, cp, M, N, K);
  } else {
    dim3 block(16, 16);
    dim3 grid(tg_ceil_div_i(N, 16), tg_ceil_div_i(M, 16));
    gemm_naive_kernel<<<grid, block, 0, TG_STREAM>>>(ap, bp, cp, M, N, K);
  }
  return out;
}
#endif  // TG_STANDALONE
