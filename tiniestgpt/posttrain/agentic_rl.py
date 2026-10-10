"""Agentic RL：对**多轮工具交互轨迹**做强化学习。

2026 年在 SWE-bench / τ-bench 上，Agentic RL 已经是 SOTA 训练范式。
它与单轮 RL 的区别不在算法，而在**轨迹的结构**：

    单轮：  prompt → 一次生成 → 一个奖励
    Agentic: prompt → [思考 → 调工具 → 观察 → 思考 → ...] → 最终奖励

由此带来三个必须处理的工程问题：

1. **损失掩码**：轨迹里有**工具返回的内容**，那不是模型生成的，
   不能对它算策略梯度（否则模型会学着"预测工具输出"）。
2. **奖励分配**：只有最终奖励时，中间每一步的贡献怎么算？
   最简单也最常见的是"最终奖励广播到所有模型生成的 token"
   （等价于把整条轨迹看成一个 action）。
3. **奖励黑客**：这段最危险——模型会学会"调用工具但忽略返回结果直接编答案"，
   或者"先假装成功再调工具修正"。必须显式检测。

本模块给出轨迹容器、掩码构造、奖励分配，以及三种 reward hacking 检测器。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

__all__ = ["Trajectory", "AgenticRLOutcome", "build_agentic_batch",
           "detect_reward_hacking", "format_reward", "tool_success_reward",
           "HackingReport"]


@dataclass
class Trajectory:
    """一条 agent 轨迹。"""

    prompt: str
    steps: List[Tuple[str, str]] = field(default_factory=list)   # [(模型输出, 工具返回)]
    final_answer: str = ""
    success: bool = False
    tool_calls: int = 0
    tokens_used: int = 0

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def text(self) -> str:
        parts = [self.prompt]
        for out, obs in self.steps:
            parts.append(out)
            if obs:
                parts.append(obs)
        parts.append(self.final_answer)
        return "\n".join(parts)


@dataclass
class AgenticRLOutcome:
    reward: float
    n_model_tokens: int
    n_masked_tokens: int
    hacking: Optional["HackingReport"] = None


def build_agentic_batch(trajs: Sequence[Trajectory],
                        reward_fn: Callable[[Trajectory], float],
                        broadcast: bool = True) -> Dict[str, torch.Tensor]:
    """把轨迹打包成可训练的 batch。

    :return: ``{"rewards":[G], "mask_ratio": float, "tokens": int}``
             —— mask 本身在 ``sft.py`` 的 ``build_sft_batch`` 里按 doc 边界构造，
             这里只给出**统计与奖励**，避免重复实现一套 token 化。
    """
    rewards = torch.tensor([reward_fn(t) for t in trajs], dtype=torch.float32)
    model_tokens = sum(t.tokens_used for t in trajs)
    obs_tokens = sum(len(obs) for t in trajs for _out, obs in t.steps)
    return {
        "rewards": rewards,
        "tokens": model_tokens,
        "mask_ratio": obs_tokens / max(obs_tokens + model_tokens, 1),
        "broadcast": bool(broadcast),
    }


# --------------------------------------------------------------------------- #
@dataclass
class HackingReport:
    ignores_tool_output: bool = False
    fakes_success_then_fixes: bool = False
    reward_loops: bool = False
    detail: str = ""

    @property
    def suspicious(self) -> bool:
        return self.ignores_tool_output or self.fakes_success_then_fixes or self.reward_loops

    def report(self) -> str:
        return f"HackingReport(可疑={self.suspicious}) {self.detail}"


def detect_reward_hacking(traj: Trajectory, max_repeat: int = 3) -> HackingReport:
    """三种最常见的 reward hacking 模式检测。"""
    r = HackingReport()
    # 1) 调了工具但输出里没引用工具返回（直接编答案）
    if traj.steps:
        outs = " ".join(o for o, _ in traj.steps)
        last_obs = traj.steps[-1][1] or ""
        if last_obs and len(last_obs) > 20 and last_obs[:20] not in outs:
            r.ignores_tool_output = True
            r.detail += "未在输出中引用最后一次工具返回；"
    # 2) 先说"完成/成功"，之后又继续调工具
    for i, (out, _obs) in enumerate(traj.steps[:-1]):
        if any(k in out.lower() for k in ("已完成", "成功", "done", "success", "finished")):
            r.fakes_success_then_fixes = True
            r.detail += f"第 {i} 步宣称成功但仍在继续；"
            break
    # 3) 同一句话重复出现（奖励循环）
    outs_list = [o for o, _ in traj.steps]
    if outs_list and len(set(outs_list)) < len(outs_list) / max(max_repeat, 1):
        r.reward_loops = True
        r.detail += "重复输出；"
    return r


def format_reward(completion: str, require_json: bool = True) -> float:
    """格式奖励：输出是否符合要求（这是防止"答对但格式错"的最廉价手段）。"""
    s = completion.strip()
    if not s:
        return 0.0
    if require_json:
        return 1.0 if (s.startswith("{") and s.endswith("}")) else 0.0
    return 1.0


def tool_success_reward(traj: Trajectory) -> float:
    """工具成功奖励：以"最终是否成功"为主，叠加格式与步数惩罚。"""
    r = 1.0 if traj.success else 0.0
    r += 0.1 * format_reward(traj.final_answer, require_json=False)
    # 步数惩罚：鼓励更短的路径（省 token 就是省钱）
    r -= 0.01 * max(traj.n_steps - 3, 0)
    return max(r, 0.0)
