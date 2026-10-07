// ---------------------------------------------------------------------------
// 02 · Reduce 三连（路线 1.3 的核心实验）
//
// 同一个数学目标（求一个数组的和），三种实现，逐级消除瓶颈：
//
//   v0 原子加      ：每个线程把值直接 atomicAdd 到全局内存的一个标量上。
//                    ✅ 简单  ❌ 所有线程抢一个地址，串行化严重，且结果不唯一（浮点加法不满足结合律）。
//   v1 共享内存树形：先在 block 内用共享内存做 **树形归约**（log2(256)=8 步），
//                    每个 block 只 atomicAdd 一次 → 全局原子操作从 N 次降到 N/512 次。
//                    ❌ 仍有 __syncthreads() 与共享内存读写。
//   v2 Warp Shuffle：warp 内用 ``__shfl_down_sync`` 走**寄存器**交换，完全不碰共享内存；
//                    只剩 warp 之间的 8 个（或更少）中间值需要走 shared。
//                    ✅ 共享内存流量降到 1/32，访存指令数大幅下降。
//
// 三个版本都要求输入是 1D 连续 float 张量。
// ---------------------------------------------------------------------------

#define TG_REDUCE_BLOCK 256

// ------------------------------- v0 --------------------------------------- //
__global__ void reduce_v0_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(out, in[i]);
  }
}

// ------------------------------- v1 --------------------------------------- //
__global__ void reduce_v1_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
  extern __shared__ float sdata[];
  const unsigned int tid = threadIdx.x;
  // 每个线程先吃掉 2 个元素（"第一步就减半"），减少一半的 block 数
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  float sum = 0.f;
  if (i < n) sum = in[i];
  if (i + blockDim.x < n) sum += in[i + blockDim.x];
  sdata[tid] = sum;
  __syncthreads();

  // 树形归约：stride 折半，注意循环内必须同步，否则会读到"别人还没写完"的值
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(out, sdata[0]);
}

// ------------------------------- v2 --------------------------------------- //
__global__ void reduce_v2_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
  extern __shared__ float sdata[];           // 只存"每个 warp 的小计"，长度 = warps/block
  const int tid = threadIdx.x;
  const int lane = tid & 31;                 // warp 内编号 0..31
  const int wid = tid >> 5;                  // warp 编号

  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  float sum = 0.f;
  if (i < n) sum = in[i];
  if (i + blockDim.x < n) sum += in[i + blockDim.x];

  // ① warp 内：全在寄存器里做，零共享内存
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }

  // ② warp 间：只有 lane0 写共享内存
  if (lane == 0) sdata[wid] = sum;
  __syncthreads();

  // ③ 第一个 warp 把每个 warp 的小计再 shuffle 归约一次
  const int nwarps = blockDim.x >> 5;
  float total = (tid < nwarps) ? sdata[tid] : 0.f;
  if (wid == 0) {
    for (int offset = nwarps / 2; offset > 0; offset >>= 1) {
      total += __shfl_down_sync(0xffffffffu, total, offset);
    }
    if (tid == 0) atomicAdd(out, total);
  }
}

// ---------------------------------------------------------------------------
#ifndef TG_STANDALONE
torch::Tensor reduce_sum(torch::Tensor x, int64_t version) {
  TORCH_CHECK(x.device().is_cuda(), "reduce_sum 需要 CUDA 张量");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "reduce_sum 目前只支持 float32");
  TORCH_CHECK(version >= 0 && version <= 2, "version 只能是 0/1/2");

  const int n = static_cast<int>(x.numel());
  auto out = torch::zeros({}, x.options());      // 0-dim 标量，原子加的累加器
  if (n == 0) return out;

  const int block = TG_REDUCE_BLOCK;
  float* out_ptr = out.data_ptr<float>();

  if (version == 0) {
    const int grid = tg_ceil_div_i(n, block);
    reduce_v0_kernel<<<grid, block, 0, TG_STREAM>>>(x.data_ptr<float>(), out_ptr, n);
  } else if (version == 1) {
    const int grid = tg_ceil_div_i(n, block * 2);
    reduce_v1_kernel<<<grid, block, block * sizeof(float), TG_STREAM>>>(
        x.data_ptr<float>(), out_ptr, n);
  } else {
    const int grid = tg_ceil_div_i(n, block * 2);
    const int smem = (block / 32) * static_cast<int>(sizeof(float));
    reduce_v2_kernel<<<grid, block, smem, TG_STREAM>>>(        x.data_ptr<float>(), out_ptr, n);
  }
  return out;
}
#endif  // TG_STANDALONE
