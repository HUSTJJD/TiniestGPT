"""导出 checkpoint 为 HF LLaMA 兼容格式（可被 transformers / vLLM 直接加载）。

    uv run python scripts/export_hf.py --checkpoint out/tiny/last.pt \
        --tokenizer data/tokenizer.json --out-dir out/tiny_hf
"""

from __future__ import annotations

import argparse

from tiniestgpt.inference.export_hf import export_hf


def main() -> None:
    ap = argparse.ArgumentParser("export_hf")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--tokenizer", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="float32",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--strict", action="store_true",
                    help="模型含 LLaMA 无法表达的特性时直接报错（不做有损导出）")
    a = ap.parse_args()
    export_hf(a.checkpoint, a.out_dir, tokenizer_path=a.tokenizer, dtype=a.dtype,
              strict=a.strict)


if __name__ == "__main__":
    main()
