"""分词器测试：训练、编解码往返、特殊 token、压缩率。"""

import numpy as np

from tiniestgpt.data.tokenizer import BPE, Tokenizer
from tiniestgpt.data.tokenizer.pretokenize import pretokenize


def test_pretokenize_basic():
    toks = pretokenize("Hello world! It's 12345 test.")
    assert "".join(toks) == "Hello world! It's 12345 test."
    # 数字按 ≤3 位切分（GPT-2 规则）
    assert any(t.isdigit() and len(t) <= 3 for t in toks)
    # 缩写单独成块
    assert "'s" in toks


def test_bpe_train_and_roundtrip():
    corpus = [
        "The little cat sat on the mat and the little dog ran away." * 20,
        "Once upon a time there was a brave knight in a small town." * 20,
        "She sells sea shells by the sea shore, and he reads books." * 20,
    ]
    bpe = BPE.train(corpus, vocab_size=400, min_pair_freq=2)
    tok = Tokenizer(bpe)
    assert tok.vocab_size >= 256 + len(tok.bpe.special_tokens)
    assert bpe.n_merges > 50

    for s in ["The little cat sat on the mat.", "Once upon a time", "hello world 123"]:
        assert tok.decode(tok.encode(s)) == s, s

    ok, total = tok.check_roundtrip(corpus)
    assert ok == total


def test_bpe_compression():
    corpus = ["the quick brown fox jumps over the lazy dog. " * 100]
    tok = Tokenizer(BPE.train(corpus, vocab_size=512, min_pair_freq=2))
    ids = tok.encode(corpus[0])
    n_bytes = len(corpus[0].encode())
    # 训练过的 BPE 应该有明显的压缩效果（>1.5 字节/token）
    assert n_bytes / len(ids) > 1.5, n_bytes / len(ids)


def test_special_tokens():
    corpus = ["hello hello hello world world world " * 50]
    tok = Tokenizer(BPE.train(corpus, vocab_size=300, min_pair_freq=2,
                              special_tokens=("<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>", "<|sep|>")))
    ids = tok.encode("A<|sep|>B", allowed_special={"<|sep|>"})
    assert tok.bpe.special_tokens["<|sep|>"] in ids
    # 未允许的特殊 token 会被当作普通文本
    ids2 = tok.encode("A<|sep|>B", allowed_special=set())
    assert tok.bpe.special_tokens["<|sep|>"] not in ids2


def test_save_load(tmp_path):
    corpus = ["abc abc abd " * 100]
    tok = Tokenizer(BPE.train(corpus, vocab_size=300, min_pair_freq=2))
    p = tmp_path / "tok.json"
    tok.save(p)
    tok2 = Tokenizer.load(p)
    s = "abc abd abc"
    assert tok.encode(s) == tok2.encode(s)
    assert tok.decode(tok.encode(s)) == s
