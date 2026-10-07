// ---------------------------------------------------------------------------
// pybind11 绑定。
//
// 我们自己写 ``PYBIND11_MODULE`` 而不是用 ``load_inline(functions=[...])``，
// 因为所有 kernel 分散在多个 .cu 文件里，统一在这里导出更清晰。
// ---------------------------------------------------------------------------
#ifndef TG_STANDALONE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "TiniestGPT 手写 CUDA 内核（教学用）";
  m.def("vector_add", &vector_add, "逐元素加法 c = a + b");
  m.def("reduce_sum", &reduce_sum, "归约求和，version=0 原子加 / 1 共享内存树形 / 2 warp shuffle");
  m.def("gemm", &gemm, "矩阵乘 [M,K]x[K,N]，tiled=1 用共享内存分块");
  m.def("softmax", &softmax, "行 softmax，online=1 用 online normalizer（单趟）");
  m.def("transpose", &transpose, "矩阵转置 [H,W]->[W,H]，padded=1 消除 bank conflict");
}
#endif  // TG_STANDALONE
