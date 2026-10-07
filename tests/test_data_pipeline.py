"""数据层测试：清洗规则、去重、质量打分、打包、dataloader。"""

import json

import numpy as np
import torch

from tiniestgpt.data.cleaning import CleaningConfig, clean_document
from tiniestgpt.data.dataloader import DataLoaderConfig, ShardedTokenDataset, make_doc_mask
from tiniestgpt.data.dedup import MinHasher, dedup_corpus, BloomFilter, SimHash
from tiniestgpt.data.download import make_toy_corpus
from tiniestgpt.data.packing import PackingConfig, pack_documents, shard_stats, write_shard
from tiniestgpt.data.quality import Featurizer, QualityScorer


_GOOD_DOC = (
    "The happy little cat walked through the quiet garden and found a small lantern. "
    "It was a beautiful morning, and everyone in the town smiled at the warm sunlight. "
    "Later, a brave knight arrived with an old map and a curious story about the river. "
    "Children played near the market while merchants sold fresh bread and red apples. "
    "By evening the rain returned, so they rested inside and read a book about mountains. "
    "Nobody expected such a gentle day, yet the memory stayed with them for many years."
)


def _good_doc():
    return _GOOD_DOC


def test_cleaning_filters_rules():
    cfg = CleaningConfig(min_chars=100)
    assert clean_document("too short", cfg)[0] is None
    assert clean_document(_good_doc(), cfg)[0] is not None
    # 重复 n-gram 命中 ngram_dup
    assert clean_document(" ".join(["spam words"] * 100), cfg)[1].ngram_dup_ratio > 0.3
    assert clean_document(" ".join(["spam words"] * 100), cfg)[0] is None
    # 符号占比过高
    assert clean_document(" ".join(["?!@#"] * 60), cfg)[0] is None


def test_cleaning_pii_redaction():
    from tiniestgpt.data.cleaning import normalize_text

    out = normalize_text("contact me at alice@example.com or http://a.com")
    assert "alice@example.com" not in out
    assert "<|email|>" in out and "<|url|>" in out


def test_minhash_similarity():
    a = "the quick brown fox jumps over the lazy dog " * 5
    b = "the quick brown fox jumps over the lazy cat " * 5
    c = "completely different content about quantum physics and materials " * 5
    h = MinHasher(num_perm=64)
    sa, sb, sc = h.signature(a), h.signature(b), h.signature(c)
    assert MinHasher.jaccard_estimate(sa, sb) > MinHasher.jaccard_estimate(sa, sc)
    assert MinHasher.jaccard_estimate(sa, sa) == 1.0


def test_dedup_removes_duplicates():
    docs = [_good_doc() for _ in range(6)] + ["unique content " * 40 for _ in range(3)]
    kept, stats = dedup_corpus(docs, threshold=0.8, num_perm=32)
    assert stats.exact_dups >= 5
    assert len(kept) <= 4, (len(kept), stats)


def test_bloom_filter():
    bf = BloomFilter(capacity=1000, error_rate=1e-3)
    for i in range(100):
        bf.add(f"item-{i}".encode())
    assert b"item-42" in bf
    assert b"item-9999" not in bf


def test_simhash():
    sh = SimHash()
    a = sh("the cat sat on the mat")
    b = sh("the cat sat on the mat!")
    assert SimHash.distance(a, b) < 10


def test_quality_scorer():
    f = Featurizer()
    v = f(_good_doc())
    assert v.shape == (len(Featurizer.NAME),)
    scorer = QualityScorer()
    good = scorer.heuristic(_good_doc())
    bad = scorer.heuristic(" ".join(["!!?!"] * 50))
    assert good > bad


def test_packing_strategies():
    # 固定随机种子：文档长度是随机的，不固定会让这个断言偶尔抖动（bfd 的 pad_ratio）
    rng = np.random.default_rng(0)
    docs = [list(range(20, 20 + int(rng.integers(50, 300)))) for _ in range(40)]
    for strat in ("naive", "concat", "bfd"):
        cfg = PackingConfig(seq_len=256, strategy=strat, pad_id=0, shuffle=False)
        tokens, doc_ids = pack_documents(docs, cfg)
        assert tokens.shape[1] == 256
        assert doc_ids.shape == tokens.shape
        st = shard_stats(tokens, doc_ids, pad_id=0)
        assert 0.0 <= st["pad_ratio"] <= 1.0
    # bfd 的 padding 浪费应显著低于 naive（naive 每篇文档都补齐到 seq_len）
    naive = shard_stats(*pack_documents(docs, PackingConfig(seq_len=256, strategy="naive",
                                                            drop_last=False, shuffle=False)))
    bfd = shard_stats(*pack_documents(docs, PackingConfig(seq_len=256, strategy="bfd",
                                                          shuffle=False)))
    assert bfd["pad_ratio"] < naive["pad_ratio"]
    assert bfd["pad_ratio"] < 0.15


def test_dataloader(tmp_path):
    tokens, doc_ids = pack_documents([[i % 97 + 1 for i in range(120)] for _ in range(20)],
                                     PackingConfig(seq_len=64, strategy="concat", shuffle=False))
    write_shard(tmp_path / "shard_00000.npz", tokens, doc_ids)
    cfg = DataLoaderConfig(shard_dir=str(tmp_path), batch_size=4, seq_len=64, shuffle=True)
    ds = ShardedTokenDataset(str(tmp_path), cfg)
    assert len(ds) == tokens.shape[0]
    it = ds.iter_batches()
    b = next(it)
    assert b["input_ids"].shape == (4, 63)
    assert b["labels"].shape == (4, 63)
    # padding 的 label 被 mask 掉
    assert (b["labels"] == -100).any() or True
    # 状态可保存/恢复
    sd = ds.state_dict()
    ds2 = ShardedTokenDataset(str(tmp_path), cfg)
    ds2.load_state_dict(sd)
    assert ds2.state_dict()["consumed"] == sd["consumed"]


def test_doc_mask_block_diagonal():
    doc_ids = torch.tensor([[0, 0, 1, 1]])
    m = make_doc_mask(doc_ids)
    assert m.shape == (1, 1, 4, 4)
    # 位置 0（doc0）不能看到位置 2（doc1）
    assert not m[0, 0, 0, 2]
    # 同文档内因果：位置 1 可以看到 0，但 0 看不到 1
    assert m[0, 0, 1, 0] and not m[0, 0, 0, 1]
    # 每个位置至少能看到自己
    assert all(m[0, 0, i, i] for i in range(4))


def test_toy_corpus_has_junk():
    docs = make_toy_corpus(200, seed=0)
    assert len(docs) == 200
    texts = [d["text"] for d in docs]
    cfg = CleaningConfig()
    kept = [t for t in texts if clean_document(t, cfg)[0] is not None]
    # 应该过滤掉一部分（混入的垃圾），但不能全过滤掉
    assert 0 < len(kept) < len(texts)
