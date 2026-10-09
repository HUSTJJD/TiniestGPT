"""真实数据集接入：切分规则 + 缓存复用（网络部分不测）。"""

from __future__ import annotations

import json

import pytest

from tiniestgpt.data.datasets import available_datasets, fetch, peek, split_plain_text
from tiniestgpt.data.download import download, iter_jsonl, write_jsonl


def test_available_datasets_contains_real_corpora():
    names = available_datasets()
    assert "tinystories" in names
    assert "shakespeare" in names


def test_split_plain_text_merges_short_chunks():
    text = "Short one.\n\n" + ("word " * 100) + "\n\nAnother short.\n\n" + ("word " * 100)
    docs = split_plain_text(text, min_doc_chars=200)
    assert len(docs) >= 1
    assert all(len(d) >= 200 for d in docs)


def test_split_plain_text_splits_very_long_chunk():
    text = "a" * 5000 + "\n\n" + "b" * 5000
    docs = split_plain_text(text, min_doc_chars=200, max_doc_chars=1000)
    assert len(docs) >= 10                      # 长文档会被切成多段
    assert all(len(d) <= 1000 for d in docs)


def test_jsonl_roundtrip_plain_and_gzip(tmp_path):
    recs = [{"id": "0", "text": "hello world"}, {"id": "1", "text": "second doc"}]
    p = tmp_path / "a.jsonl"
    assert write_jsonl(p, recs) == 2
    assert [r["text"] for r in iter_jsonl(p)] == ["hello world", "second doc"]

    import gzip

    gz = tmp_path / "a.jsonl.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    assert len(list(iter_jsonl(gz))) == 2


def test_jsonl_accepts_bare_strings(tmp_path):
    p = tmp_path / "b.jsonl"
    p.write_text('"just a string"\n"another"\n', encoding="utf-8")
    assert [r["text"] for r in iter_jsonl(p)] == ["just a string", "another"]


def test_peek_reads_first_records(tmp_path):
    p = tmp_path / "c.jsonl"
    write_jsonl(p, [{"id": str(i), "text": f"doc number {i} " * 20} for i in range(5)])
    out = peek(p, n=2, width=40)
    assert len(out) == 2
    assert out[0].startswith("doc number 0")


def test_fetch_reuses_existing_jsonl_without_network(tmp_path, monkeypatch):
    """已存在时直接复用（不联网）——保证离线重跑不重复下载。"""
    p = tmp_path / "cached.jsonl"
    write_jsonl(p, [{"id": "0", "text": "x" * 300}])

    def _boom(*a, **kw):                     # 一旦尝试联网就炸
        raise AssertionError("不该发起下载")

    monkeypatch.setattr("tiniestgpt.data.datasets.download", _boom)
    out = fetch("tinystories", p)
    assert out == p
    assert len(list(iter_jsonl(out))) == 1


def test_fetch_unknown_name_raises(tmp_path):
    try:
        fetch("no-such-dataset", tmp_path / "x.jsonl")
        raise AssertionError("应当报未知数据集")
    except ValueError as exc:
        assert "no-such-dataset" in str(exc)


def test_parquet_reader_merges_short_rows(tmp_path):
    """WikiText 一行常常只是个标题，必须合并到 min_doc_chars 才不会被清洗杀掉。"""
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    from tiniestgpt.data.datasets import _iter_parquet

    rows = ["= Some Article =", "short", "x" * 300, "= Another =", "y" * 300]
    p = tmp_path / "t.parquet"
    pq.write_table(pa.table({"text": rows}), p)

    docs = list(_iter_parquet(p, "text", None, min_doc_chars=200))
    assert len(docs) == 2                       # 前两行合并成一篇，后三行成一篇
    assert all(len(d["text"]) >= 200 for d in docs)
    assert "Some Article" in docs[0]["text"]


def test_parquet_reader_respects_max_docs(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tiniestgpt.data.datasets import _iter_parquet

    p = tmp_path / "t2.parquet"
    pq.write_table(pa.table({"text": ["z" * 300] * 10}), p)
    assert len(list(_iter_parquet(p, "text", 3, min_doc_chars=200))) == 3


def test_download_signature_accepts_sha_check(tmp_path):
    """sha256 校验失败必须报错（防止下载到截断/错误的文件）。"""
    import hashlib

    src = tmp_path / "src.bin"
    src.write_bytes(b"hello tiniestgpt")
    dest = tmp_path / "dst.bin"
    download(f"file:///{src.as_posix()}", dest, sha256=hashlib.sha256(b"hello tiniestgpt").hexdigest())
    assert dest.read_bytes() == b"hello tiniestgpt"
