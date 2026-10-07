"""OpenAI 兼容的 HTTP 服务（FastAPI）。

提供三个端点：
  * ``POST /v1/completions``       —— 文本补全
  * ``POST /v1/chat/completions``  —— 对话补全（支持 SSE 流式）
  * ``GET  /v1/models`` / ``/stats`` —— 模型与引擎运行状态

之所以做成 OpenAI 兼容：生态里的客户端（SDK、前端、Agent 框架）都能直接接进来，
这也是 vLLM / SGLang / LMDeploy 等行业实现的共同选择。
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

__all__ = ["create_app", "run_server"]


def create_app(engine, model_name: str = "tiniestgpt"):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover
        raise ImportError("服务化需要 fastapi / pydantic：pip install fastapi uvicorn pydantic") from exc

    from .sampler import SamplingParams

    app = FastAPI(title="TiniestGPT Inference Server", version="0.1.0")

    class CompletionRequest(BaseModel):
        prompt: str = ""
        max_tokens: int = 64
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = 0
        min_p: float = 0.0
        repetition_penalty: float = 1.0
        stream: bool = False
        seed: Optional[int] = None
        n: int = 1

    class ChatMessage(BaseModel):
        role: str = "user"
        content: str = ""

    class ChatCompletionRequest(BaseModel):
        messages: List[ChatMessage] = Field(default_factory=list)
        max_tokens: int = 64
        temperature: float = 1.0
        top_p: float = 1.0
        stream: bool = False
        seed: Optional[int] = None

    def _params(req: Any) -> SamplingParams:
        return SamplingParams(
            temperature=getattr(req, "temperature", 1.0),
            top_p=getattr(req, "top_p", 1.0),
            top_k=getattr(req, "top_k", 0),
            min_p=getattr(req, "min_p", 0.0),
            repetition_penalty=getattr(req, "repetition_penalty", 1.0),
            max_tokens=getattr(req, "max_tokens", 64),
            seed=getattr(req, "seed", None),
        )

    def _result_to_dict(out, prompt: str) -> Dict[str, Any]:
        return {
            "id": f"cmpl-{out.seq_id}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "text": out.text,
                "index": 0,
                "finish_reason": "stop" if out.finished else "length",
            }],
            "usage": {
                "prompt_tokens": out.prompt_tokens,
                "completion_tokens": out.completion_tokens,
                "total_tokens": out.prompt_tokens + out.completion_tokens,
            },
        }

    # ------------------------------------------------------------------ #
    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name}

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.get("/stats")
    def stats():
        return engine.stats()

    @app.get("/metrics")
    def metrics():
        """Prometheus 文本格式指标（可由 Prometheus / Grafana 直接抓取）。"""
        from fastapi.responses import PlainTextResponse

        from .metrics import default_registry

        default_registry.update_from_engine(engine)
        return PlainTextResponse(default_registry.render(),
                                 media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        if not req.prompt:
            raise HTTPException(400, "prompt 不能为空")
        outs = engine.generate(req.prompt, _params(req))
        if not outs:
            raise HTTPException(500, "生成失败")
        return _result_to_dict(outs[0], req.prompt)

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages) + "\nassistant:"
        params = _params(req)
        if req.stream:
            def gen():
                outs = engine.generate(prompt, params)
                text = outs[0].text if outs else ""
                for chunk in text.split():
                    payload = {"id": "chatcmpl-0", "object": "chat.completion.chunk",
                               "created": int(time.time()), "model": model_name,
                               "choices": [{"index": 0, "delta": {"content": chunk + " "},
                                            "finish_reason": None}]}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        outs = engine.generate(prompt, params)
        out = outs[0] if outs else None
        text = out.text if out else ""
        return {
            "id": "chatcmpl-0", "object": "chat.completion", "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": out.prompt_tokens if out else 0,
                      "completion_tokens": out.completion_tokens if out else 0,
                      "total_tokens": (out.prompt_tokens + out.completion_tokens) if out else 0},
        }

    return app


def run_server(engine, host: str = "127.0.0.1", port: int = 8000,
               model_name: str = "tiniestgpt") -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要 uvicorn：pip install uvicorn") from exc
    app = create_app(engine, model_name=model_name)
    uvicorn.run(app, host=host, port=port, log_level="info")
