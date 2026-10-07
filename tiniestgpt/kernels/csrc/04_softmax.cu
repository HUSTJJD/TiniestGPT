// ---------------------------------------------------------------------------
// 04 · Softmax：naive 与 online normalizer
//
// 朴素写法要好几趟：先求 max（防溢出），再求 sum，最后除。
// 每一趟都要把整行重新读一遍 → 3 次 HBM 读 + 1 次写。
//
// **Online softmax**（Milakov & Gimelshein 2018，也是 FlashAttention 的核心）
// 只扫一遍：边扫边维护 (running_max m, running_sum s)，遇到新元素 v 就做
//      m_new = max(m, v)
//      s_new = s · exp(m - m_new) + exp(v - m_new)
// 于是 HBM 读降到 1 次。这就是"FlashAttention 为什么快"的最本质解释。
//
// 多线程合并时也有一个 online 版本的归约公式（下面 tree 合并用的）：
//      m = max(m1, m2)
//      s = s1·exp(m1 - m) + s2·exp(m2 - m)
// ---------------------------------------------------------------------------

// ------------------------------ 朴素版 ------------------------------------ //
__global__ void softmax_naive_kernel(const float* __restrict__ in,
                                     float* __restrict__ out, int rows, int cols) {
  __shared__ float s_max[TG_SOFTMAX_BLOCK];
  __shared__ float s_sum[TG_SOFTMAX_BLOCK];

  const int row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x;
  const float* x = in + (size_t)row * (size_t)cols;
  float* y = out + (size_t)row * (size_t)cols;

  // ① max
  float m = -FLT_MAX;
  for (int i = tid; i < cols; i += TG_SOFTMAX_BLOCK) m = fmaxf(m, x[i]);
  s_max[tid] = m;
  __syncthreads();
  for (int s = TG_SOFTMAX_BLOCK / 2; s > 0; s >>= 1) {
    if (tid < s) s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
    __syncthreads();
  }
  m = s_max[0];

  // ② sum
  float sum = 0.f;
  for (int i = tid; i < cols; i += TG_SOFTMAX_BLOCK) sum += expf(x[i] - m);
  s_sum[tid] = sum;
  __syncthreads();
  for (int s = TG_SOFTMAX_BLOCK / 2; s > 0; s >>= 1) {
    if (tid < s) s_sum[tid] += s_sum[tid + s];
    __syncthreads();
  }

  // ③ 归一化
  const float inv = 1.f / fmaxf(s_sum[0], 1e-20f);
  for (int i = tid; i < cols; i += TG_SOFTMAX_BLOCK) {
    y[i] = expf(x[i] - m) * inv;
  }
}

// --------------------------- online 版（单趟） ---------------------------- //
__global__ void softmax_online_kernel(const float* __restrict__ in,
                                      float* __restrict__ out, int rows, int cols) {
  __shared__ float s_m[TG_SOFTMAX_BLOCK];
  __shared__ float s_s[TG_SOFTMAX_BLOCK];

  const int row = blockIdx.x;
  if (row >= rows) return;
  const int tid = threadIdx.x;
  const float* x = in + (size_t)row * (size_t)cols;
  float* y = out + (size_t)row * (size_t)cols;

  float m = -FLT_MAX;
  float s = 0.f;
  for (int i = tid; i < cols; i += TG_SOFTMAX_BLOCK) {
    const float v = x[i];
    const float m_new = fmaxf(m, v);
    s = s * expf(m - m_new) + expf(v - m_new);
    m = m_new;
  }

  // 用"online 合并公式"做 block 内树形归约（不能直接 sum，因为各线程基线不同）
  s_m[tid] = m;
  s_s[tid] = s;
  __syncthreads();
  for (int off = TG_SOFTMAX_BLOCK / 2; off > 0; off >>= 1) {
    if (tid < off) {
      const float m1 = s_m[tid], s1 = s_s[tid];
      const float m2 = s_m[tid + off], s2 = s_s[tid + off];
      const float m_new = fmaxf(m1, m2);
      s_m[tid] = m_new;
      s_s[tid] = s1 * expf(m1 - m_new) + s2 * expf(m2 - m_new);
    }
    __syncthreads();
  }

  m = s_m[0];
  const float inv = 1.f / fmaxf(s_s[0], 1e-20f);
  for (int i = tid; i < cols; i += TG_SOFTMAX_BLOCK) {
    y[i] = expf(x[i] - m) * inv;
  }
}

// ---------------------------------------------------------------------------
#ifndef TG_STANDALONE
torch::Tensor softmax(torch::Tensor x, int64_t online) {
  TORCH_CHECK(x.device().is_cuda(), "softmax 需要 CUDA 张量");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "softmax 目前只支持 float32");
  TORCH_CHECK(x.dim() == 2, "softmax 只支持 2D [rows, cols]");

  const int rows = static_cast<int>(x.size(0));
  const int cols = static_cast<int>(x.size(1));
  auto out = torch::empty_like(x);
  if (rows == 0 || cols == 0) return out;

  const int block = TG_SOFTMAX_BLOCK;
  if (online) {
    softmax_online_kernel<<<rows, block, 0, TG_STREAM>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), rows, cols);
  } else {
    softmax_naive_kernel<<<rows, block, 0, TG_STREAM>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), rows, cols);
  }
  return out;
}
#endif  // TG_STANDALONE
