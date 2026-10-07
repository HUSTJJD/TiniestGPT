# 08 · CUDA 编程与算子优化

> 对应 AIInfraGuide 路线**第一层**。
> 代码在 `tiniestgpt/kernels/`：`csrc/` 是 `.cu` 源码，`cuda_ops.py` 是 Python 封装
> （每个算子都带 PyTorch 参考实现，无 GPU 时自动降级）。

---

## 0. 为什么要补这一层

前面几篇文档里的所有优化（PagedAttention、量化、投机解码、CUDA Graph）
**最终都要落到 kernel 上**。只会调 PyTorch 算子的人，看到 Nsight 报告里
"memory throughput 92%" 也不知道下一步该怎么办。

这一层的核心一句话：

> **"内存访问模式决定运行速度"**，而不是"算得多就慢"。

---

## 1. GPU 硬件：先搞清楚工厂长什么样

把一块 GPU 想象成一座有几千个**只会做加减乘除**的工人的超级工厂：

| 概念 | 类比 | 关键点 |
|---|---|---|
| SM（Streaming Multiprocessor） | 车间 | A100 108 个 / H100 132 个；一个 block 只会被调度到一个 SM 上 |
| CUDA Core | 工人 | 每个时钟做 1 次 FP32 FMA |
| Tensor Core | 特种设备 | 一条指令算一个 16×16×16 矩阵乘，这才是大模型算力的主要来源 |
| Warp | 班（32 人） | **最小调度单位**：GPU 下发指令以 warp 为单位，所以 `blockDim` 必须是 32 的倍数 |
| HBM | 园区外的仓库 | 带宽 ~3.35 TB/s（H100），但延迟是共享内存的 ~100 倍 |

### Memory Wall：为什么带宽比算力先崩

RTX 3060：FP32 算力 12.7 TFLOP/s，带宽 360 GB/s。
一次 4 字节的加载能撑多少次乘加？`12.7e12 / (360e9/4) ≈ 141` 次。
也就是说：**一个数据如果只用不到 141 次，你的 kernel 就是 memory bound。**

朴素 GEMM 里每个元素只用 1 次 → 必然被带宽卡死 → 这就是为什么必须做 tiling。

### 存储层次（快 → 慢）

```
寄存器 (1 cycle)  >  共享内存/L1 (~20 cycle)  >  L2  >  HBM (~600 cycle)  >  主机内存
```

写 kernel 的全部艺术，就是**把数据尽可能久地留在快的那层里复用**。

---

## 2. 五个 kernel 实验（按目录顺序读）

| 文件 | 实验 | 路线对应 |
|---|---|---|
| `csrc/01_vector_add.cu` | 线程索引、边界检查、异步 launch | 1.4 |
| `csrc/02_reduce.cu` | **Reduce 三连** | 1.3 "Reduce 三连" |
| `csrc/03_gemm.cu` | 朴素 vs 共享内存分块，对比 cuBLAS | 1.3 "GEMM 分块" |
| `csrc/04_softmax.cu` | 朴素 vs **online normalizer** | FlashAttention 的前置知识 |
| `csrc/05_transpose.cu` | **Bank Conflict** 直觉（32×32 转置） | 1.3 "Bank Conflict 直觉" |

跑起来：

```bash
# 环境自检
python -c "from tiniestgpt.kernels import cuda_ops as ck; print(ck.status())"

# 基准（Reduce 三连 / GEMM vs cuBLAS / Softmax）
python benchmarks/cuda_kernels.py
python benchmarks/cuda_kernels.py --reduce-n 4194304 --gemm 1024 1024 1024
```

数值一致性由 `tests/test_cuda_kernels.py` 保证（CUDA 不可用时整体 skip）。

### 2.1 Reduce 三连到底省了什么

| 版本 | 全局原子操作次数 | 共享内存流量 | 省在哪 |
|---|---|---|---|
| v0 `atomicAdd` | N | 0 | — （所有线程抢同一个地址，严重串行化，且浮点结果不唯一） |
| v1 共享内存树形 | N / 512 | block 内 log₂(256)=8 轮 | 原子操作从 N 降到 N/512 |
| v2 warp shuffle | N / 512 | 只剩 warp 间的小计（1/32） | warp 内走**寄存器**（`__shfl_down_sync`），完全不碰 shared |

关键代码（`02_reduce.cu`）：

```cpp
// warp 内：寄存器交换，零共享内存
for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
if (lane == 0) sdata[wid] = sum;   // 只有 lane0 写 shared
```

> ⚠️ 树形归约里 `__syncthreads()` 必须在**循环体内**，
> 否则会读到"别人还没写完"的值——这是最常见的 reduce bug。

### 2.2 GEMM：算术强度的提升

朴素版：算 C 的一个元素要读 A 的一整行 + B 的一整列，
算术强度 `2K flops / 2K loads ≈ 1`。

分块版：把 A、B 切成 `TILE×TILE` 搬进共享内存，一个 block 负责 C 的一个子块，
每块数据被复用 `TILE` 次 → HBM 访存降到 `O(MNK / TILE)`。

再往上还有四步（本项目留作练习，注释里写了方向）：

1. 每线程算 4×4 子块（提高寄存器复用）
2. `float4` 128-bit 向量化访存
3. 双缓冲 / `cp.async`（Ampere+）
4. **换 Tensor Core**（`wmma` / `mma`）← cuBLAS 快的根本原因

### 2.3 Online Softmax = FlashAttention 的核心

朴素 softmax 要扫三趟（max → sum → 除），online 版只扫一趟：

```
m_new = max(m, v)
s_new = s · exp(m - m_new) + exp(v - m_new)
```

多线程归并时也要用 online 版本的合并公式（直接 `sum` 是错的，因为各线程基线不同）：

```
m = max(m1, m2)
s = s1 · exp(m1 - m) + s2 · exp(m2 - m)
```

**理解了这段，FlashAttention 就只剩"外层循环遍历 KV 块、内层遍历 Q 块"这一层窗户纸了。**

### 2.4 Bank Conflict：加一列为什么快 30 倍

共享内存分 32 个 bank（每 bank 4 字节）。转置时：

- 读 `tile[ty][tx]`：线程按 `tx` 变 → 连续地址 → 无冲突
- 写 `tile[tx][ty]`：行宽正好 32 → **32 个线程全落到同一个 bank** → 32-way conflict

修法只有一行：

```cpp
__shared__ float tile[32][32 + 1];   // padding 一列，每行错开一个 bank
```

---

## 3. Occupancy 与资源分配

```
Occupancy = 活跃 warp 数 / SM 支持的最大 warp 数
```

受三个硬约束限制（取最小值）：

1. **warp 数上限**（通常 64/SM）
2. **寄存器**：`regs_per_thread × threads ≤ 65536/SM`
3. **共享内存**：`smem_per_block × blocks ≤ 228KB/SM`（H100）

直觉：occupancy 高 ≠ 一定快。memory-bound kernel 需要高 occupancy 来**隐藏延迟**；
compute-bound kernel 更依赖 ILP（指令级并行）。Nsight Compute 的
`Launch Statistics` + `Occupancy` 面板会直接告诉你卡在哪个约束上。

---

## 4. 性能分析工具链

| 工具 | 回答什么问题 | 用法 |
|---|---|---|
| `torch.profiler` | 哪一步慢、有没有 CPU-GPU 空洞 | 见 `tiniestgpt/common/profiler.py::torch_profile` |
| Nsight Systems | GPU idle gap 是 CPU 预处理、通信等待还是 launch 开销 | `nsys profile -o out python -m tiniestgpt.cli pretrain ...` |
| Nsight Compute | 单个 kernel 是 memory bound 还是 compute bound | `ncu --set full -o out python benchmarks/cuda_kernels.py` |

读 Nsight Compute 报告的顺序：

1. **`Speed Of Light` 面板**：先看 Compute 与 Memory 两条，谁接近 100% 谁就是瓶颈
2. **`Memory Workload Analysis`**：L1/L2 命中率、HBM 吞吐
3. **`Scheduler / Warp State`**：张 `Stall` 原因（Long Scoreboard = 等显存）
4. **`Occupancy`**：是被寄存器还是共享内存限制了

> 我们的 `.cu` 编译时带了 `-lineinfo`，所以 Nsight 能下钻到源码行。

### 常见结论速查

| 现象 | 诊断 | 处方 |
|---|---|---|
| SOL Memory 高、Compute 低 | memory bound | 提高数据复用（tiling / 融合 / 向量化访存） |
| SOL 双低、Launch 里 grid 很小 | 并行度不足 | 提高 occupancy 或改算法 |
| `Stall Long Scoreboard` 高 | 等全局内存 | 预取 / 换共享内存 / 提高 occupancy 隐藏延迟 |
| 大量小 kernel 连续 launch | launch overhead | **算子融合** / CUDA Graph |
| Bank Conflicts > 0 | 共享内存访问模式 | 加 padding 或改索引映射 |

---

## 5. 检验标准（自测清单）

对照 AIInfraGuide 路线 1.3，跑完这一层你应该能：

- [ ] 不查资料说出你手上那块卡的 HBM 容量 / 带宽 / 共享内存上限量级，
      并解释"为什么带宽往往先成为瓶颈"
- [ ] 给定 2GB 梯度，估算 NVLink(900GB/s) vs PCIe Gen5(64GB/s) 的 AllReduce 耗时差
- [ ] 独立写出 Reduce 三连，并用 `benchmarks/cuda_kernels.py` 说明每步省在哪
- [ ] 手动构造并消除 32×32 转置的 bank conflict
- [ ] 写出共享内存分块 GEMM，在 1024³ 上达到 cuBLAS 的 50% 以上
- [ ] 白板推导 FlashAttention 的 tiling + online softmax，说清 HBM 读写
      为什么从 O(N²) 降到 O(N)
- [ ] 用 Nsight Systems 抓一次训练 iteration，指出 GPU idle gap 的来源；
      用 Nsight Compute 判断某个 kernel 是 memory bound 还是 compute bound

---

## 6. 环境要求与排错

| 条件 | 说明 |
|---|---|
| GPU | 任意 NVIDIA 显卡（本项目在 RTX 3060 / sm_86 验证） |
| CUDA Toolkit | **必须**（`nvcc` 要在 PATH 里）。torch 自带的 CUDA **runtime** 不等于 nvcc |
| Windows | 还需要 MSVC（JIT 用它编译 host 端） |
| Ninja | torch 的 C++/CUDA JIT 用 ninja 驱动（`uv sync --extra kernel` 已带上） |

三者缺一，`cuda_ops.status()` 会明确告诉你缺哪个，
并且**所有算子自动回退到 PyTorch 实现**，不会让程序崩掉：

```bash
python -c "from tiniestgpt.kernels import cuda_ops as ck; print(ck.status())"
```

### 6.1 三条验证路径（按"能跑通的概率"排序）

| 路径 | 命令 | 依赖 |
|---|---|---|
| ① 独立自检（最稳） | `python scripts/verify_cuda_kernels.py` | 只要 nvcc + 任意 MSVC/g++ |
| ② JIT 扩展（日常使用） | `python benchmarks/cuda_kernels.py` | nvcc + ninja + **与 torch 头文件兼容的** MSVC |
| ③ 单元测试 | `uv run pytest tests/test_cuda_kernels.py` | 同 ②；不满足时自动 skip |

路径 ① 用 `-DTG_STANDALONE` 编译——`csrc/*.cu` 里的 kernel 本身**不依赖 PyTorch**，
所以可以单独编出一个可执行程序做数值自检（也能顺手看出自己的 GPU 型号）。

### 6.2 Windows 上的版本地狱（以及我们怎么绕过）

nvcc 对 host 编译器有**白名单**（`crt/host_config.h` 里写着
`#if _MSC_VER < 1910 || _MSC_VER >= 1940`），而新版 Visual Studio（VS 2026，MSVC 14.5x）
已经超出白名单；反过来，降级到很旧的 MSVC（14.29）又会因为 torch 头文件用了新语法而编译失败。

`loader.py` 的处理顺序：

1. `vswhere` 找到 VS 安装目录；
2. 解析 nvcc 的 `host_config.h` 得到允许的 `_MSC_VER` 上限；
3. 在 `VC/Tools/MSVC/` 下挑一个**满足上限且最新**的工具集（`-vcvars_ver=...`）；
4. 若全都超上限，就选最新的并加 `-allow-unsupported-compiler`；
5. 仍然失败 → `status()["error"]` 给出原因，全部算子走 PyTorch 参考实现。

可用环境变量覆盖：`TINIESTGPT_MSVC_TOOLSET=14.44`、`TINIESTGPT_NVCC_ALLOW_UNSUPPORTED=1`。

> 想一步到位：装 **CUDA Toolkit ≥ 12.4**（推荐 12.8，与 torch cu128 对齐）+ VS 2022。
