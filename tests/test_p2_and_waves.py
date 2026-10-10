"""P2 与 Waves 4/5/6：约束解码、调度策略、低比特、LoRA、进阶解码、
PPO/偏好族/Agentic RL/PRM、Workspace/护栏/A2A、数据专项/Agent 评测。"""

import torch
import torch.nn as nn

# ---------------- 推理 P2 ---------------- #
from tiniestgpt.inference.grammar import (ConstrainedDecoder, from_json_schema,
                                          from_literal, from_regex)
from tiniestgpt.inference.policy import (CostLedger, FairScheduler, RequestCost,
                                         SemanticRouter)
from tiniestgpt.inference.lowbit import compare_formats, nvfp4_quantize, micro_dequantize
from tiniestgpt.inference.lora import LoRAConfig, LoRAPool, inject_lora
from tiniestgpt.inference.advanced_decode import (beam_search, build_draft_tree,
                                                  TreeSpecConfig, verify_tree, TreeSpecStats)


def test_dfa_regex_quantifiers_and_alternation():
    d = from_regex("ab*c")
    assert d.accepts("ac") and d.accepts("abc") and d.accepts("abbc")
    assert not d.accepts("ab") and not d.accepts("abd")
    alt = from_regex("a|bc")
    assert alt.accepts("a") and alt.accepts("bc") and not alt.accepts("b")


def test_dfa_schema_and_masking():
    num = from_json_schema({"type": "number"})
    assert num.accepts("12") and num.accepts("-3") and num.accepts("1.5")
    assert not num.accepts("abc")
    dec = ConstrainedDecoder(from_literal(["yes", "no"]), alphabet=list("yesno"))
    assert dec.feed("y") and not dec.done
    assert dec.feed("e") and dec.feed("s") and dec.done


def test_fair_scheduler_prefers_high_priority_but_ages():
    fs = FairScheduler()
    fs.enqueue(RequestCost("low", prompt_tokens=100, priority=0))
    fs.enqueue(RequestCost("high", prompt_tokens=4000, priority=5))
    assert fs.pop_next().rid == "high"
    # 低优先级等待过久后应被老化机制捞起
    fs2 = FairScheduler()
    fs2.enqueue(RequestCost("low", prompt_tokens=100, priority=0))
    fs2.queue["low"].enqueue_at -= 100        # 人为把入队时间往前挪
    fs2.enqueue(RequestCost("high", prompt_tokens=100, priority=1))
    assert fs2.pop_next().rid == "low"


def test_semantic_router_and_cost_ledger():
    r = SemanticRouter()
    assert r.route("def f(): pass").rid == "coder"
    assert r.route("hi").rid == "general"
    led = CostLedger()
    c = led.record(RequestCost("a", prompt_tokens=1000, output_tokens=2000))
    assert abs(c - (1.0 * 0.001 + 2.0 * 0.002)) < 1e-9
    assert led.summary()["requests"] == 1.0


def test_lowbit_nvfp4_beats_nothing_and_is_reasonable():
    w = torch.randn(64, 128)
    q, s = nvfp4_quantize(w)
    d = micro_dequantize(q, s, 16, w.shape)
    rel = float((d - w).abs().mean() / w.abs().mean())
    assert rel < 0.15, f"NVFP4 误差过大: {rel}"
    errs = compare_formats(w)
    assert set(errs) == {"nvfp4", "mxfp4", "int4"}
    assert all(v > 0 for v in errs.values())


def test_lora_inject_and_pool_eviction():
    m = nn.Sequential(nn.Linear(16, 16))
    rep = inject_lora(m, LoRAConfig(r=4, targets=("0",)))
    assert len(rep) == 1
    assert m(torch.randn(2, 16)).shape == (2, 16)
    pool = LoRAPool(max_adapters=2)
    for i in range(3):
        pool.add(f"a{i}", torch.zeros(4, 16), torch.zeros(16, 4), 1.0)
    assert len(pool.adapters) == 2 and pool.evicted == 1


def test_tree_spec_and_beam_search():
    torch.manual_seed(0)
    dl = [torch.randn(32) for _ in range(4)]
    edges, tok = build_draft_tree(dl, TreeSpecConfig(n_draft=6))
    assert len(edges) > 0
    st = TreeSpecStats()
    chain = verify_tree(torch.stack(dl), edges, st)
    assert len(chain) <= len(edges)
    assert 0.0 <= st.acceptance_rate <= 1.0

    def step(_t):
        return torch.randn(32)
    beams = beam_search(step, [1, 2], num_beams=3, max_new_tokens=4)
    assert len(beams) == 3
    assert all(len(b.tokens) >= 2 for b in beams)


# ---------------- 后训练 ---------------- #
from tiniestgpt.posttrain.ppo import PPOConfig, Critic, compute_gae, ppo_loss
from tiniestgpt.posttrain.preference import (ipo_loss, kto_loss, orpo_loss,
                                             rejection_sample, simpo_loss)
from tiniestgpt.posttrain.agentic_rl import (Trajectory, detect_reward_hacking,
                                             tool_success_reward)
from tiniestgpt.posttrain.advanced_reward import (ProcessRewardModel, PRMConfig,
                                                  RewardShaping, ThinkingBudget,
                                                  aggregate_prm, rloo_advantages)


def test_gae_and_ppo_loss():
    r = torch.rand(2, 5)
    v = torch.rand(2, 5)
    dones = torch.zeros(2, 5)
    dones[:, -1] = 1
    adv, ret = compute_gae(r, v, dones)
    assert adv.shape == r.shape and torch.isfinite(adv).all()
    old = torch.randn(2, 5)
    out = ppo_loss(old + 0.1 * torch.randn(2, 5), old, adv, PPOConfig())
    assert torch.isfinite(out["loss"])
    assert 0.0 <= float(out["clip_frac"]) <= 1.0


def test_critic_shape():
    assert Critic(64)(torch.randn(2, 3, 64)).shape == (2, 3)


def test_preference_family_losses_are_finite():
    torch.manual_seed(0)
    pc, pr = torch.randn(4), torch.randn(4)
    rc, rr = torch.randn(4), torch.randn(4)
    assert torch.isfinite(ipo_loss(pc, pr, rc, rr))
    assert torch.isfinite(kto_loss(pc, rc, torch.tensor([1.0, 0.0, 1.0, 0.0])))
    assert torch.isfinite(orpo_loss(torch.rand(4), pc, pr))
    assert torch.isfinite(simpo_loss(pc, pr, torch.full((4,), 10.0), torch.full((4,), 20.0)))


def test_simpo_is_length_invariant():
    """SimPO 的核心是**长度归一化**：同等"每 token 质量"下，长度翻倍损失不变。

    这正是它治长度偏差的机制——DPO 的隐式奖励与长度正相关，
    于是模型学会"写更长"来刷分；归一化之后这条路被堵死。
    """
    short = simpo_loss(torch.tensor([-10.0]), torch.tensor([-20.0]),
                       torch.tensor([20.0]), torch.tensor([20.0]))
    long = simpo_loss(torch.tensor([-20.0]), torch.tensor([-40.0]),
                      torch.tensor([40.0]), torch.tensor([40.0]))
    assert abs(float(short - long)) < 1e-6, f"长度归一化失效: {float(short)} vs {float(long)}"


def test_rejection_sampling_and_best_of_n():
    from tiniestgpt.posttrain.preference import best_of_n
    cands = ["7", "8", "9"]

    def sampler(_p):
        return cands

    kept = rejection_sample(["q"], sampler, lambda _p, c: c == "7")
    assert kept == [("q", "7")]
    assert best_of_n("q", sampler, lambda _p, c: float(c))[0] == "9"


def test_agentic_rl_hacking_detection():
    good = Trajectory("q", [("调用计算器", "结果是42")], final_answer="42", success=True)
    assert not detect_reward_hacking(good).suspicious
    bad = Trajectory("q", [("已完成", "结果42"), ("再改改", "")],
                     final_answer="42", success=True)
    rep = detect_reward_hacking(bad)
    assert rep.suspicious, "宣称成功后又继续调用工具，应被标记"
    assert tool_success_reward(good) > 0


def test_prm_and_rloo_and_shaping():
    prm = ProcessRewardModel(PRMConfig(dim=16))
    s = prm(torch.randn(2, 3, 16))
    assert s.shape == (2, 3) and ((s >= 0) & (s <= 1)).all()
    assert float(aggregate_prm(s, "min").max()) <= 1.0
    adv = rloo_advantages(torch.tensor([1.0, 0.0, 1.0, 0.0]), 2)
    assert float(adv.abs().sum()) > 0
    sh = RewardShaping(length_penalty=0.01, target_len=4)
    assert sh.apply(1.0, "a b c d e f") < 1.0, "超长应被扣分"
    assert ThinkingBudget().budget_for("high") == 4096


# ---------------- Agent ---------------- #
from tiniestgpt.agent.workspace import SkillLibrary, TaskState, Workspace
from tiniestgpt.agent.guardrails import Budget, CostGuard, HITL, LoopDetector, Permission
from tiniestgpt.agent.a2a import A2ARegistry, AgentCard, GraphMemory


def test_workspace_persists_and_resumes(tmp_path):
    ws = Workspace(tmp_path / "ws")
    ws.write("a.txt", "hello")
    assert ws.read("a.txt") == "hello" and ws.ls() == ["a.txt"]
    ws.save_state(TaskState("t", step=7, data={"k": 1}))
    assert Workspace(tmp_path / "ws").resume("t").step == 7


def test_skill_library_retrieve_and_persist(tmp_path):
    lib = SkillLibrary(str(tmp_path / "skills.json"))
    lib.learn("add", "做加法", "用 calculator")
    lib.add_lesson("别用 python_repl 跑死循环")
    assert lib.retrieve("做加法")[0].name == "add"
    lib2 = SkillLibrary(str(tmp_path / "skills.json"))
    assert "add" in lib2.skills and len(lib2.lessons) == 1


def test_cost_guard_stops_on_budget_and_loops():
    # 每次用不同的 args，避免先被循环检测器拦下
    g = CostGuard(Budget(max_steps=2))
    assert g.check("think", "a").allowed
    g.consume(tokens=10)
    assert g.check("think", "b").allowed
    g.consume(tokens=10)
    assert not g.check("think", "c").allowed and "步数" in g.stopped_reason
    assert g.usage()["tokens"] == 20.0
    ld = LoopDetector(repeat_threshold=2)
    assert ld.observe("calc", "1+1") is False
    assert ld.observe("calc", "1+1") is True


def test_permission_and_hitl():
    p = Permission()
    assert p.level_of("think") == "allow"
    assert p.level_of("write_file") == "approval"
    assert p.level_of("python_sandbox") == "deny"
    h = HITL(auto_approve=True)
    assert h.gate("write_file").allowed
    assert not h.gate("python_sandbox").allowed


def test_a2a_and_graph_memory():
    reg = A2ARegistry()
    reg.register(AgentCard("coder", skills=["code"]), lambda m: f"code:{m}")
    t = reg.send("main", "coder", "write f")
    assert t.state == "completed" and t.artifacts == ["code:write f"]
    assert reg.send("main", "nobody", "x").state == "failed"

    gm = GraphMemory()
    gm.add("A", "knows", "B")
    gm.add("B", "knows", "C")
    assert gm.query(s="A") == [("A", "knows", "B")]
    assert len(gm.multi_hop("A", hops=2)) == 2


# ---------------- 数据 / 评测 ---------------- #
from tiniestgpt.data.curriculum import (AnnealingSchedule, code_quality, difficulty_score,
                                        verify_math)
from tiniestgpt.data.dedup_suffix import GlobalDeduplicator, build_suffix_array, suffix_dedup
from tiniestgpt.eval.agent_eval import AgentEvaluator, AgentTask, pass_at_k, pass_k_estimator


def test_suffix_array_and_dedup():
    sa = build_suffix_array("banana")
    assert sorted(sa) == list(range(6))
    keep, st = suffix_dedup(["hello world", "xxhello worldyy", "totally different"],
                            min_len=6)
    assert len(keep) == 2 and st.removed == 1
    assert st.longest_shared >= 6


def test_global_dedupicator_catches_cross_shard():
    g = GlobalDeduplicator(ngram=3)
    shard1 = g.add_shard(["a b c d e f g h i j k l m n o p q r s t"])
    shard2 = g.add_shard(["a b c d e f g h i j k l m n o p q r s t", "x y z"])
    assert len(shard1) == 1 and len(shard2) == 1
    assert g.stats["cross_shard_removed"] == 1


def test_annealing_and_curriculum():
    sch = AnnealingSchedule(start_frac=0.5, end_frac=1.0,
                            base_weights={"web": 0.9, "wiki": 0.1},
                            final_weights={"web": 0.1, "wiki": 0.9})
    assert sch.weights_at(0.2)["wiki"] == 0.1
    assert abs(sch.weights_at(0.75)["wiki"] - 0.5) < 1e-6
    assert difficulty_score("a") < difficulty_score("a " * 500)


def test_code_quality_and_math_verifier():
    assert code_quality("x = 1\n")["syntax_ok"] == 1.0
    assert code_quality("def (:\n")["syntax_ok"] == 0.0
    assert code_quality("# " + "\n# ".join(["c"] * 20))["score"] < 1.0
    assert verify_math("1+1", "答案是 2", "2")
    assert not verify_math("1+1", "3", "2")


def test_agent_pass_k_metrics():
    evaluator = AgentEvaluator(
        [AgentTask("t", "q", lambda a: a == "ok")], repeats=4)
    # 前两次成功、后两次失败 → pass^2 只有 1/3 组全通过
    seq = iter([True, True, False, True])

    def agent(_p):
        ok = next(seq)
        return {"answer": "ok" if ok else "bad", "steps": 2, "tokens": 100,
                "used_tool": True}

    evaluator.run(agent)
    s = evaluator.summary()
    assert abs(s["success_rate"] - 0.75) < 1e-9
    # pass^k 用**不重叠**分组：[T,T] 全过 / [F,T] 不全过 → 1/2
    assert abs(pass_k_estimator([True, True, False, True], 2) - 0.5) < 1e-9
    assert 0.0 <= pass_at_k(4, 3, 2) <= 1.0
    assert "平均成功率" in evaluator.table()
