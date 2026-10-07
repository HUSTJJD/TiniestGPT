# 数据：决定模型上限的第一环

> "数据决定上限，算法只是逼近这个上限。" —— 这条经验法则在 LLM 时代被反复验证。

## 1. 流水线总览

```
原始文本 → [清洗] → [去重] → [质量打分] → [分词] → [打包] → 分片(.npz)
              ↓         ↓          ↓
          过滤原因统计  精确/模糊  启发式 or 判别器
```

每个阶段都会把统计写进 `data/pipeline_report.json`。**调参的第一步永远是看这份报告**。

## 2. 清洗（`cleaning.py`）

工业界（C4 / RefinedWeb / Dolma / FineWeb）的共同结论：**规则过滤比模型过滤更划算**，
因为它便宜、可解释、可审计。

本项目实现的规则：

| 类别 | 规则 | 参数 |
|---|---|---|
| 归一化 | NFKC、HTML 标签/实体、控制字符、空白折叠 | `do_nfkc` / `strip_html` |
| 隐私 | 邮箱 / URL / IP / 手机 / 密钥 → `<\|xxx\|>` | `redact_pii` |
| 结构 | 过短/过长、重复行、重复 8-gram、列表页、URL 农场 | `min_chars` 等 |
| 语言学 | 字母占比、符号占比、数字占比、平均词长、脚本一致性 | `min_alpha_ratio` 等 |

**实验建议**：把 `max_ngram_dup_ratio` 从 0.30 调到 0.05，观察过滤率上升多少、
下游 BPE 的"字节/token"压缩率如何变化（重复模板会污染词表）。

## 3. 去重（`dedup.py`）

两级：

1. **精确去重**：全文 BLAKE2b 哈希 + Bloom Filter（无假阴性，可能有假阳性）。
2. **模糊去重**：MinHash + Banding LSH + Union-Find。

MinHash 的原理：``P(min h(A) == min h(B)) = Jaccard(A, B)``。
于是"签名中相同位置的比例"就是 Jaccard 的无偏估计。
用 LSH 把 128 维签名切成 `b` 个 band，每个 band 内完全相同才成为候选对，
把 O(N²) 的两两比较降到近似 O(N)。

阈值关系：``threshold ≈ (1/b)^(1/r)``，其中 `r` 是每个 band 的行数。
本项目默认 `num_perm=64, rows_per_band=8` → `b=8` → 阈值≈0.68（配合显式阈值 0.85 做二次确认）。

**实验建议**：把 `dedup_threshold` 从 0.85 降到 0.7，看保留文档数下降多少；
再检查被合并的文档是否真的语义重复（打印 verbose 日志）。

## 4. 质量打分（`quality.py`）

- **启发式**：把统计量线性组合后过 sigmoid，得到 0~1 分。可解释、零成本。
- **判别器**：纯 numpy 的逻辑回归，用"自举伪标签"（启发式分最高/最低各 20%）
  训练，比启发式更能捕捉"像不像高质量文本"。

这正是 FineWeb / QuRater 的做法的简化版：**用少量高质量种子（Wikipedia / 教科书）
训练一个小分类器，再给全量网页打分**。

## 5. 分词（`tokenizer/`）

从零实现 BPE：

- **预切分**（`pretokenize.py`）：复刻 GPT-2 正则。三个关键设计：
  缩写单独成块、数字按 ≤3 位切分、空白贴到词前。
  > 注意：空白必须跟随后一个词，否则 encode/decode 往返会丢空格。
  本项目为此专门写了测试 `tests/test_tokenizer.py::test_pretokenize_basic`。
- **训练**（`bpe.py`）：增量更新 pair 计数 + 懒删除最大堆。
  朴素实现是 O(V·N)，本项目是 O(Σ|含该 pair 的词|)，8k 词表秒级完成。
- **压缩率**：健康的英文词表应该在 3.5~5 字节/token 之间。
  太低说明词表太小（序列变长、算力浪费）；太高说明词表被噪声占满。

## 6. 打包（`packing.py`）

三种策略，对应三代做法：

| 策略 | 做法 | padding 浪费 | 跨文档污染 |
|---|---|---|---|
| `naive` | 一篇文档一条样本 | 高 | 无 |
| `concat` | 首尾相接后切开 | 几乎为零 | 有（需 doc mask） |
| `bfd` | Best-Fit-Decreasing 装箱 | 低 | 无（配块对角掩码） |

`bfd` 是现代数据管线的常用做法：既不截断任何文档，又几乎不浪费 token，
配合 `ShardedTokenDataset` 提供的 `doc_ids` 与 `make_doc_mask` 做块对角注意力，
做到"零浪费 + 无跨文档污染"。

## 7. 加载（`dataloader.py`）

- 分片用 `np.load` 内存映射，语料远大于内存也不 OOM；
- 洗牌只发生在**索引层**（零拷贝），并且可以序列化 -> 支持断点续训；
- `BatchPrefetcher` 用独立 CUDA stream + pinned memory 把 H2D 拷贝与计算重叠。

## 8. 一键复现

```bash
# 离线玩具语料（含 15% 刻意混入的垃圾，用来验证过滤规则真的生效）
python -m tiniestgpt.cli data --config recipes/data_toy.yaml

# 用自己的 JSONL
python -m tiniestgpt.cli data --source jsonl --jsonl my_corpus.jsonl --out-dir data
```
