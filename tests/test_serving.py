"""结构化输出约束解码 + Prometheus 指标 + 服务化小冒烟。"""

from __future__ import annotations

import json

import pytest
import torch

from tiniestgpt.inference.metrics import MetricsRegistry
from tiniestgpt.inference.structured import StructuredDecoder, validate_against_schema


class _Tok:
    """极简分词器：id ↔ 单字符，便于精确验证状态机。"""

    def __init__(self) -> None:
        self.alphabet = list("{}[]\":,0123456789truefalsn \tabcemnxyz-.")
        self.stoi = {c: i for i, c in enumerate(self.alphabet)}

    @property
    def size(self) -> int:
        return len(self.alphabet)

    def decode(self, ids, skip_special=True):
        # 引擎的 vocab 是几千，这里用取模把任意 id 映射到可打印字符
        return "".join(self.alphabet[i % len(self.alphabet)] for i in ids)


# --------------------------------------------------------------------------- #
# JSON 前缀状态机
# --------------------------------------------------------------------------- #
def test_prefix_accepts_valid_json():
    from tiniestgpt.inference.structured import JsonPrefix

    p = JsonPrefix()
    assert all(p.feed(c) for c in '{"a": 1, "b": [true, false, null], "c": {"d": -1.5e3}}')
    assert p.done


def test_prefix_rejects_invalid_json():
    from tiniestgpt.inference.structured import JsonPrefix

    p = JsonPrefix()
    assert p.feed("{")
    assert p.feed('"')
    assert p.feed("a")
    assert p.feed('"')
    assert not p.feed('"')          # 少了冒号，直接接字符串 → 非法


def test_prefix_rejects_trailing_garbage():
    from tiniestgpt.inference.structured import JsonPrefix

    p = JsonPrefix()
    assert all(p.feed(c) for c in '{"a":1}')
    assert p.done
    assert not p.feed("x")          # JSON 已闭合，不能再输出内容


# --------------------------------------------------------------------------- #
# StructuredDecoder
# --------------------------------------------------------------------------- #
def test_decoder_masks_invalid_tokens():
    tok = _Tok()
    dec = StructuredDecoder(tok, tok.size)
    logits = torch.zeros(tok.size)
    masked = dec.mask_logits(logits)
    # 起始状态（已经吃掉 '{'）下，只允许 key 的引号、'}' 与空白
    allowed = [i for i, v in enumerate(masked.tolist()) if v > float("-inf")]
    chars = {tok.alphabet[i] for i in allowed}
    assert '"' in chars and "}" in chars
    assert "1" not in chars          # 裸数字不能当 key


def test_decoder_completes_and_validates():
    tok = _Tok()
    dec = StructuredDecoder(tok, tok.size, schema={"type": "object",
                                                   "required": ["name"],
                                                   "properties": {"name": {"type": "string"}}})
    for ch in '"name":"x"}':
        tid = tok.stoi[ch]
        dec.feed_token(tid)
    assert dec.complete
    ok, msg = dec.validate()
    assert ok, msg
    assert dec.value() == {"name": "x"}


def test_decoder_schema_validation_failure():
    ok, msg = validate_against_schema({"name": 123},
                                      {"type": "object", "properties": {"name": {"type": "string"}}})
    assert not ok and "name" in msg
    ok, _ = validate_against_schema({"name": "x"},
                                    {"type": "object", "required": ["name"]})
    assert ok


def test_decoder_mask_always_allows_progress():
    """无论走到哪一步，允许的 token 都不能为空（否则会死锁）。"""
    tok = _Tok()
    dec = StructuredDecoder(tok, tok.size)
    for ch in ['"k', '"]', ":", "1", "}"]:
        for c in ch:
            if c in tok.stoi:
                dec.feed_token(tok.stoi[c])
        assert dec.allowed_token_ids()


# --------------------------------------------------------------------------- #
# Prometheus 指标
# --------------------------------------------------------------------------- #
def test_engine_structured_output_is_valid_json():
    """端到端：给引擎一个 json_schema，输出必须能被 json.loads 解析。"""
    torch.manual_seed(0)
    from tiniestgpt.inference.engine import EngineConfig, InferenceEngine
    from tiniestgpt.inference.sampler import SamplingParams
    from tiniestgpt.model.factory import build_model

    tok = _Tok()
    base = build_model("tiny")
    cfg = EngineConfig(max_num_seqs=1, block_size=16, dtype="fp32", device="cpu",
                       max_model_len=128, enable_prefix_caching=False)
    engine = InferenceEngine(build_model(base.cfg), tok, cfg)
    engine.model.load_state_dict(base.state_dict())

    schema = {"type": "object"}
    prompt = list(range(4, 20))
    sid = engine.add_request(prompt, SamplingParams(max_tokens=24, temperature=0.0),
                             json_schema=schema)
    while engine.scheduler.has_unfinished():
        engine.step()

    out = engine.outputs[sid]
    text = tok.decode(out.output_ids)

    # 约束解码的核心保证：**输出永远是合法 JSON 的前缀**（不可能出现语法错误）
    from tiniestgpt.inference.structured import JsonPrefix

    probe = JsonPrefix()
    assert probe.feed("{")
    assert all(probe.feed(c) for c in text), f"生成了非法 JSON 前缀: {text!r}"

    assert out.finished
    if probe.done:                       # 提前闭合 → 可以直接解析
        assert isinstance(json.loads("{" + text), dict)


def test_metrics_render_contains_core_metrics():
    reg = MetricsRegistry()
    reg.counter("requests_total", 3.0, "累计请求数")
    reg.gauge("kv_cache_usage_ratio", 0.42, "KV Cache 占用率")
    reg.observe_ttft(0.12)
    reg.observe_ttft(0.35)
    text = reg.render()
    assert "tiniestgpt_requests_total 3.0" in text
    assert "tiniestgpt_kv_cache_usage_ratio 0.42" in text
    assert "tiniestgpt_ttft_seconds_count 2" in text
    assert 'tiniestgpt_ttft_seconds_bucket{le="0.5"} 2' in text
    assert "# TYPE tiniestgpt_ttft_seconds histogram" in text


def test_metrics_histogram_bucket_boundaries():
    reg = MetricsRegistry(buckets=(0.1, 1.0))
    reg.observe("x", 0.05)
    reg.observe("x", 0.5)
    text = reg.render()
    assert 'tiniestgpt_x_bucket{le="0.1"} 1' in text
    assert 'tiniestgpt_x_bucket{le="1.0"} 2' in text


def test_metrics_update_from_engine_like_object():
    class FakeScheduler:
        running = [1, 2, 3]
        waiting = [4]

    class FakeEngine:
        scheduler = FakeScheduler()

        def stats(self):
            return {"block_usage": 0.5, "preempted": 7, "steps": 100, "free_blocks": 12}

    reg = MetricsRegistry()
    reg.update_from_engine(FakeEngine())
    text = reg.render()
    assert "tiniestgpt_kv_cache_usage_ratio 0.5" in text
    assert "tiniestgpt_requests_running 3" in text
    assert "tiniestgpt_requests_waiting 1" in text
    assert "tiniestgpt_preemptions_total 7.0" in text
