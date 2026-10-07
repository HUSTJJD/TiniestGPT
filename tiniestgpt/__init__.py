"""TiniestGPT: 小而全的大模型全链路实践项目。"""

__version__ = "0.1.0"

# 延迟导入，避免 import tiniestgpt 时就把 torch 拉起来（CLI/文档工具可轻量化运行）
__all__ = ["__version__"]
