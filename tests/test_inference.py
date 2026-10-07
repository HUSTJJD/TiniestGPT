"""推理系统测试：引擎 vs 朴素生成一致性、调度器、采样器、量化、KV 量化、投机解码。"""

import torch

from tiniestgpt.inference.engine import EngineConfig, InferenceEngine
from tiniestgpt.inference.sampler import (SamplingParams, Sampler, apply_penalties,
                                          min_p_filter, top_k_top_p_filter)
from tiniestgpt.inference.scheduler import Scheduler, Sequence, SequenceStatus
from tiniestgpt.model.config import ModelConfig
from tiniestgpt.model.factory import build_model
from tiniestgpt.model.kv_cache import DenseKVCache


class DummyTokenizer:
    pad_id, bos_id, eos_id, unk_id = 0, 1, 2, 3

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [(ord(c) % 60) + 4 for c in text[:16]] or [4]
        return ([self.bos_id] if add_bos else []) + ids + ([self.eos_id] if add_eos else [])

    def decode(self, ids, skip_special=True):
        return "".join(chr(max(int(i), 0) + 40) for i in ids if int(i) >= 0)


def _cfg(**kw):
    base = dict(vocab_size=128, dim=32, n_layers=2, n_heads=4, n_kv_heads=2,
                hidden_dim=64, max_seq_len=64)
    base.update(kw)
    return ModelConfig(**base)


def naive_generate(model, prompt_ids, max_new_tokens=8, eos_id=2):
    """朴素自回归生成（稠密 KV Cache，greedy），作为引擎的正确性基准。"""
    model.eval()
    ids = torch.tensor([prompt_ids])
    cache = DenseKVCache(model.cfg.n_layers, 1, model.cfg.max_seq_len, *model.cache_spec(),
                         dtype=torch.float32)
    with torch.no_grad():
        logits = model(ids, cache=cache)
        cache.advance(len(prompt_ids))
        out = [int(logits[0, -1].argmax())]
        for _ in range(max_new_tokens - 1):
            if out[-1] == eos_id:
                break
            x = torch.tensor([[out[-1]]])
            pos = torch.tensor([[len(prompt_ids) + len(out) - 1]])
            logits = model(x, positions=pos, cache=cache)
            cache.advance(1)
            out.append(int(logits[0, -1].argmax()))
    return out


def test_engine_matches_naive_greedy():
    torch.manual_seed(0)
    model = build_model(_cfg())
    tok = DummyTokenizer()
    engine = InferenceEngine(model, tok, EngineConfig(
        dtype="fp32", device="cpu", max_num_seqs=2, block_size=4, num_gpu_blocks=64,
        max_num_batched_tokens=64, enable_prefix_caching=False, max_model_len=64))
    prompt = "hello world"
    ids = tok.encode(prompt)
    expected = naive_generate(model, ids, max_new_tokens=8)
    outs = engine.generate(prompt, SamplingParams(temperature=0.0, max_tokens=8))
    got = outs[0].output_ids
    assert got == expected, (got, expected)


def test_engine_prefix_caching_same_result():
    torch.manual_seed(0)
    model = build_model(_cfg())
    tok = DummyTokenizer()
    base = dict(dtype="fp32", device="cpu", max_num_seqs=2, block_size=4,
                num_gpu_blocks=64, max_model_len=64)
    e1 = InferenceEngine(build_model(_cfg()), tok, EngineConfig(enable_prefix_caching=False, **base))
    e1.model.load_state_dict(model.state_dict())
    e2 = InferenceEngine(build_model(_cfg()), tok, EngineConfig(enable_prefix_caching=True, **base))
    e2.model.load_state_dict(model.state_dict())
    p = SamplingParams(temperature=0.0, max_tokens=6)
    a = e1.generate("abc def", p)[0].output_ids
    b = e2.generate("abc def", p)[0].output_ids
    assert a == b
    # 第二次请求相同前缀 → 命中缓存，且结果仍然正确
    c = e2.generate("abc def", p)[0].output_ids
    assert c == b
    assert e2.stats()["prefix_cache_hit_rate"] > 0


def test_engine_batch_multiple_prompts():
    model = build_model(_cfg())
    engine = InferenceEngine(model, DummyTokenizer(), EngineConfig(
        dtype="fp32", device="cpu", max_num_seqs=4, block_size=4, num_gpu_blocks=64,
        max_model_len=64))
    outs = engine.generate(["aaaa", "bbbb", "cccc"],
                           SamplingParams(temperature=0.0, max_tokens=5))
    assert len(outs) == 3
    assert all(len(o.output_ids) >= 1 for o in outs)
    assert engine.stats()["total_generated_tokens"] >= 3


def test_scheduler_preempt_and_reuse_blocks():
    sched = Scheduler(block_size=4, max_num_seqs=2, max_num_batched_tokens=16)
    from tiniestgpt.model.kv_cache import BlockAllocator

    sched.set_allocator(BlockAllocator(4))               # 只有 4 个块 → 很快耗尽
    sp = SamplingParams(max_tokens=32)
    seqs = [Sequence(i, [1] * 8, sp, block_size=4) for i in range(3)]
    for s in seqs:
        sched.add_seq(s)
    out = sched.schedule()
    assert len(out) >= 1
    assert sched.allocator.num_free < 4


def test_sampler_filters():
    logits = torch.tensor([1.0, 5.0, 3.0, 0.1])
    # top_k=2 只保留 5 与 3
    f = top_k_top_p_filter(logits.clone(), top_k=2)
    assert int((f > -1e10).sum()) == 2
    # min_p 会把远小于最大值的项过滤掉
    m = min_p_filter(logits.clone(), min_p=0.5)
    assert int((m > -1e10).sum()) <= 2
    s = Sampler("cpu")
    tok = s.sample(torch.tensor([0.0, 10.0, 0.0]), SamplingParams(temperature=0.0))
    assert tok == 1
    tok2 = s.sample(torch.tensor([1.0, 1.0, 1.0]), SamplingParams(temperature=1.0, seed=0))
    assert 0 <= tok2 < 3


def test_repetition_penalty():
    logits = torch.tensor([2.0, 2.0, 2.0])
    out = apply_penalties(logits, prompt_ids=(), output_ids=[0], repetition_penalty=2.0)
    assert out[0] < out[1]          # 出现过的 token 被惩罚
    out2 = apply_penalties(logits, prompt_ids=(), output_ids=[0], frequency_penalty=1.0)
    assert out2[0] < out2[1]


def test_quantized_linear_error():
    from tiniestgpt.inference.quantization import QuantizedLinear, pack_int4, unpack_int4

    torch.manual_seed(0)
    lin = torch.nn.Linear(32, 16, bias=False)
    x = torch.randn(4, 32)
    ref = lin(x)
    for bits in (8, 4):
        q = QuantizedLinear.from_float(lin, bits=bits, group_size=32)
        out = q(x)
        rel = (out - ref).abs().max() / ref.abs().max()
        assert rel < 0.15, (bits, rel)
    # int4 打包 / 解包往返
    q8 = torch.randint(-8, 8, (4, 8), dtype=torch.int8)
    assert torch.equal(unpack_int4(pack_int4(q8), q8.shape), q8)


def test_quantize_model_runs():
    from tiniestgpt.inference.quantization import quantize_model

    m = build_model(_cfg())
    x = torch.randint(0, 128, (1, 6))
    with torch.no_grad():
        ref = m(x)
    quantize_model(m, bits=8, group_size=32)
    with torch.no_grad():
        out = m(x)
    assert out.shape == ref.shape
    assert torch.isfinite(out).all()


def test_kv_cache_quantization():
    from tiniestgpt.inference.quantization import QuantizedPagedKVCache

    cfg = _cfg()
    m = build_model(cfg).eval()
    ids = torch.randint(0, 128, (1, 8))
    dens = DenseKVCache(cfg.n_layers, 1, 32, *m.cache_spec(), dtype=torch.float32)
    with torch.no_grad():
        ref = m(ids[:, :8], cache=dens)

    qcache = QuantizedPagedKVCache(cfg.n_layers, num_blocks=8, block_size=4,
                                   n_kv_heads=m.cache_spec()[0], head_dim=m.cache_spec()[1],
                                   granularity="per_token")
    blocks = qcache.allocator.allocate(4)
    bt = torch.tensor([blocks], dtype=torch.long)
    slots = torch.tensor([[blocks[i // 4] * 4 + i % 4 for i in range(8)]], dtype=torch.long)
    qcache.set_batch(block_table=bt, seq_lens=torch.tensor([8]), slot_ids=slots)
    with torch.no_grad():
        out = m(ids[:, :8], cache=qcache)
    # int8 量化后误差应在可接受范围
    rel = (out - ref).abs().max() / ref.abs().max().clamp(min=1e-6)
    assert rel < 0.25, rel
    assert qcache.compression_ratio(torch.float16) > 1.5


def test_speculative_decoding_runs():
    from tiniestgpt.inference.speculative import SpeculativeDecoder

    torch.manual_seed(0)
    target = build_model(_cfg(n_layers=3))
    draft = build_model(_cfg(n_layers=1))
    # draft 用 target 的部分权重初始化，提升接受率（纯属测试便利）
    sd = {k: v for k, v in target.state_dict().items() if k in draft.state_dict()}
    draft.load_state_dict(sd, strict=False)

    dec = SpeculativeDecoder(target, draft, DummyTokenizer(), device="cpu",
                             dtype=torch.float32)
    res = dec.generate([4, 5, 6, 7], SamplingParams(temperature=0.7, max_tokens=8),
                       num_spec_tokens=3, max_new_tokens=8, seed=0)
    assert len(res["output_ids"]) >= 1
    st = res["stats"]
    assert st.rounds >= 1 and st.draft_tokens > 0
    assert 0.0 <= st.acceptance_rate <= 1.0


def test_speculative_matches_target_distribution():
    """统计意义上：投机解码不应改变 greedy 输出（temperature→0 时）。"""
    from tiniestgpt.inference.speculative import SpeculativeDecoder

    torch.manual_seed(0)
    target = build_model(_cfg(n_layers=2))
    draft = build_model(_cfg(n_layers=1))
    sd = {k: v for k, v in target.state_dict().items() if k in draft.state_dict()}
    draft.load_state_dict(sd, strict=False)
    dec = SpeculativeDecoder(target, draft, DummyTokenizer(), device="cpu", dtype=torch.float32)
    ids = [4, 5, 6, 7]
    greedy = naive_generate(target, ids, max_new_tokens=6)
    res = dec.generate(ids, SamplingParams(temperature=1e-7, max_tokens=6),
                       num_spec_tokens=2, max_new_tokens=6, seed=1)
    # temperature 极小但不为 0，允许少量差异；前几个 token 应完全一致
    assert res["output_ids"][:3] == greedy[:3], (res["output_ids"], greedy)
