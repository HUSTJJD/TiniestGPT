# 数据：决定模型上限的第一环

> "数据决定上限，算法只是逼近这个上限。" —— 这条经验法则在 LLM 时代被反复验证。

## 1. 流水线总览

```
原始文本 → [清洗] → [去重] → [质量打分] → [分词] → [打包] → 分片(.npz)
              ↓         ↓          ↓
          过滤原因统计  精确/模糊  启发式 or 判别器
```

每个阶段都会把统计写进 `data/pipeline_report.json`。**调参的第一步永远是看这份报告**。

## 1.1 用真实数据集（`datasets.py`）

玩具语料能验证**流水线对不对**，但验证不了**模型能不能真的学到东西**——
模板合成语料的分布太窄，loss 掉得再低也只是在背模板。内置了几个可直接下载的
真实语料（只需联网一次，之后走缓存）：

```bash
python -m tiniestgpt.cli datasets --probe     # 列出内置数据集并探测远端体积（不下载）
```

| name | 体积 | 来源 / 地位 |
|---|---|---|
| `tinystories` | 19.4 MB | TinyStories **valid** split（Eldan & Li, 2023）—— 默认用它 |
| `tinystories_train` | 1.9 GB | TinyStories **train** split —— 方法上更规范，用 `max_docs` 截断 |
| `wikitext2` | 6.4 MB | WikiText-2 raw train —— **标准 LM benchmark**（parquet，需 pyarrow） |
| `wikitext103` | 157 MB ×2 | WikiText-103 raw train —— 业界标准 benchmark |
| `shakespeare` | 1.1 MB | Karpathy char-rnn 的莎士比亚全集，纯字符级规律 |

关于"标不标准"，说清楚三件事：

1. **TinyStories 是真实的公开学术数据集**，但它定位是"研究小模型语言能力"的合成式语料，
   不是主流预训练 benchmark（C4 / OpenWebText / FineWeb / The Pile 那一挂）。
   用它训出来的 ppl **不能**和论文里的数字横向对比 —— 词表、分词、口径全都不同。
2. **我默认用的是 valid split，这在方法上是不严谨的**：正确做法是 train 训练 / valid 评估。
   之所以默认 valid，只是因为它 19 MB、本机几十秒能跑完；
   想规范就把 `source_name` 改成 `tinystories_train`（1.9 GB，用 `max_docs` 截断到你要的量）。
3. **想和论文对标就用 WikiText**：`recipes/data_wikitext2.yaml` 已经配好，
   口径一致时 perplexity 可以比（但仍要注意词表大小与 token 化方式的影响）。

```bash
# 一条命令：下载 → 切文档 → 清洗/去重/打分 → 训练 BPE → 打包
python -m tiniestgpt.cli data --config recipes/data_tinystories.yaml
python -m tiniestgpt.cli pretrain --config recipes/pretrain_tinystories.yaml

# 标准 benchmark
python -m tiniestgpt.cli data --config recipes/data_wikitext2.yaml
```

WikiText-2 实测：13,014 篇 → 2.70M tokens（4.01 B/token，padding 2.5%），全流程 98 s。

自己接数据源也很简单：准备一个每行 `{"id": ..., "text": ...}` 的 JSONL，然后

```yaml
source: jsonl
source_path: data/raw/mine.jsonl
```

**本机实测（RTX 3060 12GB）**：

| 阶段 | 结果 |
|---|---|
| 下载 + 切分 | 19.4 MB → 12,553 篇 |
| 清洗 / 去重 / 打分 | 12,552 篇（几乎无重复） |
| BPE（vocab 4096） | 8.1 s，压缩率 **3.99 B/token** |
| 打包（seq_len 512, bfd） | 9,535 packs / 4.88M tokens，**padding 仅 4.1%** |
| 全流程 | **66 s** |
| 预训练（49M 参数，300 步） | loss 2.87 → 2.37，eval ppl **9.5**，11.5k tok/s，显存 8.3 GB |

几个真实语料上才会踩到的点：

* **词表必须调大**：2048 → 4096，否则 token 数膨胀、压缩率变差；
* **清洗阈值要放宽**：真实文本不像模板那样"标准"，先跑一遍看
  `reject_reasons` 分布再收紧，别一上来就卡死；
* **滑动窗口要关掉**（`attn_window: -1`）：故事里前后指代很常见；
* `data/packed/shard_00000.npz` 会被覆盖，玩具语料与真实语料**共用同一目录**，
  切换数据源时记得重跑 `data` 流水线。

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
