// ---------------------------------------------------------------------------
// TG_STANDALONE 模式下的自检 main（由 scripts/verify_cuda_kernels.py 编译执行）。
//
// 存在的意义：JIT 编译需要"torch 头文件 ↔ nvcc ↔ MSVC"三者版本互相兼容，
// 这在某些机器上很难同时满足（例如 CUDA 12.3 + VS 2026）。
// 这条路径**完全不依赖 PyTorch**，只验证 kernel 本身的数值正确性。
// ---------------------------------------------------------------------------
#ifndef TG_STANDALONE
#error "standalone_main.cu 只能在 -DTG_STANDALONE 下编译"
#endif

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static int g_failures = 0;

static void report(const char* name, double max_err, double tol) {
  bool ok = max_err <= tol;
  if (!ok) g_failures++;
  printf("  [%s] %-28s max_err=%.3e (tol=%.1e)\n", ok ? "PASS" : "FAIL", name, max_err, tol);
}

// --------------------------------------------------------------------------- //
static void test_vector_add() {
  const int n = 100000;
  std::vector<float> a(n), b(n), c(n);
  for (int i = 0; i < n; ++i) { a[i] = (float)i * 0.001f; b[i] = (float)(n - i) * 0.002f; }

  float *da, *db, *dc;
  TG_CUDA_CHECK(cudaMalloc(&da, n * sizeof(float)));
  TG_CUDA_CHECK(cudaMalloc(&db, n * sizeof(float)));
  TG_CUDA_CHECK(cudaMalloc(&dc, n * sizeof(float)));
  TG_CUDA_CHECK(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  TG_CUDA_CHECK(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  const int block = 256;
  vector_add_kernel<<<tg_ceil_div_i(n, block), block>>>(da, db, dc, n);
  TG_CUDA_CHECK(cudaMemcpy(c.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));

  double err = 0.0;
  for (int i = 0; i < n; ++i) err = fmax(err, fabs((double)c[i] - (double)(a[i] + b[i])));
  report("vector_add", err, 1e-6);
  cudaFree(da); cudaFree(db); cudaFree(dc);
}

// --------------------------------------------------------------------------- //
static void test_reduce(int version) {
  const int n = 1 << 20;
  std::vector<float> x(n);
  double ref = 0.0;
  for (int i = 0; i < n; ++i) { x[i] = (float)((i % 97) - 48) * 0.01f; ref += x[i]; }

  float *dx, *dout;
  TG_CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
  TG_CUDA_CHECK(cudaMalloc(&dout, sizeof(float)));
  TG_CUDA_CHECK(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  TG_CUDA_CHECK(cudaMemset(dout, 0, sizeof(float)));

  const int block = 256;
  if (version == 0) {
    reduce_v0_kernel<<<tg_ceil_div_i(n, block), block>>>(dx, dout, n);
  } else if (version == 1) {
    reduce_v1_kernel<<<tg_ceil_div_i(n, block * 2), block, block * sizeof(float)>>>(dx, dout, n);
  } else {
    reduce_v2_kernel<<<tg_ceil_div_i(n, block * 2), block, (block / 32) * sizeof(float)>>>(dx, dout, n);
  }
  float got = 0.f;
  TG_CUDA_CHECK(cudaMemcpy(&got, dout, sizeof(float), cudaMemcpyDeviceToHost));

  char name[64];
  snprintf(name, sizeof(name), "reduce_sum v%d", version);
  // v0 把 N 个数依次累加到**同一个** float 累加器上，误差会随 N 增长
  // （1e6 个元素的相对误差约 1e-4）——它不只是慢，还更不准，这正是优化的理由之一。
  double tol = (version == 0) ? 1e-3 : 1e-4;
  report(name, fabs((double)got - ref) / fmax(1.0, fabs(ref)), tol);
  cudaFree(dx); cudaFree(dout);
}

// --------------------------------------------------------------------------- //
static void test_gemm(bool tiled) {
  const int M = 97, K = 128, N = 53;
  std::vector<float> a(M * K), b(K * N), c(M * N), ref(M * N, 0.f);
  for (int i = 0; i < M * K; ++i) a[i] = (float)((i % 31) - 15) * 0.05f;
  for (int i = 0; i < K * N; ++i) b[i] = (float)((i % 17) - 8) * 0.05f;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float s = 0.f;
      for (int k = 0; k < K; ++k) s += a[m * K + k] * b[k * N + n];
      ref[m * N + n] = s;
    }

  float *da, *db, *dc;
  size_t sz_a = (size_t)M * K * sizeof(float), sz_b = (size_t)K * N * sizeof(float);
  size_t sz_c = (size_t)M * N * sizeof(float);
  TG_CUDA_CHECK(cudaMalloc(&da, sz_a));
  TG_CUDA_CHECK(cudaMalloc(&db, sz_b));
  TG_CUDA_CHECK(cudaMalloc(&dc, sz_c));
  TG_CUDA_CHECK(cudaMemcpy(da, a.data(), sz_a, cudaMemcpyHostToDevice));
  TG_CUDA_CHECK(cudaMemcpy(db, b.data(), sz_b, cudaMemcpyHostToDevice));

  if (tiled) {
    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid(tg_ceil_div_i(N, TILE), tg_ceil_div_i(M, TILE));
    gemm_tiled_kernel<TILE><<<grid, block>>>(da, db, dc, M, N, K);
  } else {
    dim3 block(16, 16);
    dim3 grid(tg_ceil_div_i(N, 16), tg_ceil_div_i(M, 16));
    gemm_naive_kernel<<<grid, block>>>(da, db, dc, M, N, K);
  }
  TG_CUDA_CHECK(cudaMemcpy(c.data(), dc, sz_c, cudaMemcpyDeviceToHost));

  double err = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) err = fmax(err, fabs((double)c[i] - (double)ref[i]));
  report(tiled ? "gemm tiled" : "gemm naive", err, 1e-3);
  cudaFree(da); cudaFree(db); cudaFree(dc);
}

// --------------------------------------------------------------------------- //
static void test_softmax(bool online) {
  const int rows = 32, cols = 1000;
  std::vector<float> x(rows * cols), y(rows * cols), ref(rows * cols);
  for (int i = 0; i < rows * cols; ++i) x[i] = (float)((i % 211) - 105) * 0.07f;
  for (int r = 0; r < rows; ++r) {
    float m = -FLT_MAX;
    for (int c = 0; c < cols; ++c) m = fmax(m, x[r * cols + c]);
    double s = 0.0;
    for (int c = 0; c < cols; ++c) s += exp((double)x[r * cols + c] - m);
    for (int c = 0; c < cols; ++c)
      ref[r * cols + c] = (float)(exp((double)x[r * cols + c] - m) / s);
  }

  float *dx, *dy;
  size_t sz = (size_t)rows * cols * sizeof(float);
  TG_CUDA_CHECK(cudaMalloc(&dx, sz));
  TG_CUDA_CHECK(cudaMalloc(&dy, sz));
  TG_CUDA_CHECK(cudaMemcpy(dx, x.data(), sz, cudaMemcpyHostToDevice));
  if (online) {
    softmax_online_kernel<<<rows, TG_SOFTMAX_BLOCK>>>(dx, dy, rows, cols);
  } else {
    softmax_naive_kernel<<<rows, TG_SOFTMAX_BLOCK>>>(dx, dy, rows, cols);
  }
  TG_CUDA_CHECK(cudaMemcpy(y.data(), dy, sz, cudaMemcpyDeviceToHost));

  double err = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) err = fmax(err, fabs((double)y[i] - (double)ref[i]));
  report(online ? "softmax online" : "softmax naive", err, 1e-5);
  cudaFree(dx); cudaFree(dy);
}

// --------------------------------------------------------------------------- //
static void test_transpose(bool padded) {
  const int H = 70, W = 130;
  std::vector<float> x(H * W), y(H * W);
  for (int i = 0; i < H * W; ++i) x[i] = (float)i;

  float *dx, *dy;
  size_t sz = (size_t)H * W * sizeof(float);
  TG_CUDA_CHECK(cudaMalloc(&dx, sz));
  TG_CUDA_CHECK(cudaMalloc(&dy, sz));
  TG_CUDA_CHECK(cudaMemcpy(dx, x.data(), sz, cudaMemcpyHostToDevice));

  dim3 block(TG_TRANSPOSE_TILE, TG_TRANSPOSE_TILE);
  dim3 grid(tg_ceil_div_i(W, TG_TRANSPOSE_TILE), tg_ceil_div_i(H, TG_TRANSPOSE_TILE));
  if (padded) {
    transpose_padded_kernel<<<grid, block>>>(dx, dy, W, H);
  } else {
    transpose_naive_kernel<<<grid, block>>>(dx, dy, W, H);
  }
  TG_CUDA_CHECK(cudaMemcpy(y.data(), dy, sz, cudaMemcpyDeviceToHost));

  // out 的形状是 [W, H]：out[j*H + i] 应等于 x[i*W + j]
  double err = 0.0;
  for (int i = 0; i < H; ++i)
    for (int j = 0; j < W; ++j)
      err = fmax(err, fabs((double)y[(size_t)j * H + i] - (double)x[(size_t)i * W + j]));
  report(padded ? "transpose padded" : "transpose naive", err, 0.0);
  cudaFree(dx); cudaFree(dy);
}

// --------------------------------------------------------------------------- //
int main() {
  printf("TiniestGPT CUDA kernels - standalone self-check\n");
  int dev = 0;
  cudaDeviceProp prop{};
  TG_CUDA_CHECK(cudaGetDevice(&dev));
  TG_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("device: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

  test_vector_add();
  test_reduce(0);
  test_reduce(1);
  test_reduce(2);
  test_gemm(false);
  test_gemm(true);
  test_softmax(false);
  test_softmax(true);
  test_transpose(false);
  test_transpose(true);

  printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "ALL PASSED",
         g_failures, g_failures == 1 ? "" : "s");
  return g_failures ? 1 : 0;
}
