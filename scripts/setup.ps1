# TiniestGPT 环境初始化（Windows / PowerShell）
#   .\scripts\setup.ps1
#
# 做四件事：
#   1. 用 uv 创建虚拟环境（Python 3.14）
#   2. 从 PyTorch 官方索引装 **CUDA 版** torch（PyPI 上的默认是 CPU 版！）
#   3. 安装 serving + dev 依赖，并以 editable 方式安装本项目
#   4. 验证 CUDA / Triton 可用性

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

Write-Host "[1/4] 创建 uv 虚拟环境 ..." -ForegroundColor Cyan

# --- 关键 workaround -------------------------------------------------------
# uv 在 Windows 上为 console script 生成 .exe trampoline 时，若 TEMP 指向 8.3 短名
# 路径（形如 C:\Users\DAVIDS~1\AppData\Local\Temp），会报：
#     Failed to update Windows PE resources ... (os error -2147024786)
# 这里把 TEMP/TMP 强制指向一个"长名"目录即可解决。
$tmpDir = Join-Path $ProjectRoot ".tmp"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
$env:TEMP = (Resolve-Path $tmpDir).Path
$env:TMP  = (Resolve-Path $tmpDir).Path
# 项目目录与 uv 缓存通常不在同一文件系统（D: vs C:），硬链接会退化，直接指定 copy
$env:UV_LINK_MODE = "copy"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "未找到 uv，请先安装：  irm https://astral.sh/uv/install.ps1 | iex" -ForegroundColor Red
    exit 1
}

uv venv --python 3.14
if ($LASTEXITCODE -ne 0) { throw "uv venv 失败" }

Write-Host "[2/4] 安装依赖（PyTorch cu128，约 2.6 GB，首次较慢）..." -ForegroundColor Cyan
uv sync --extra all
if ($LASTEXITCODE -ne 0) { throw "uv sync 失败（若卡住无输出，是正在下载 torch，请耐心等待）" }

Write-Host "[3/4] 验证 ..." -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" "scripts\verify_env.py"
if ($LASTEXITCODE -ne 0) { throw "环境验证失败" }

Write-Host "[4/4] 完成。使用方式：" -ForegroundColor Green
Write-Host "     .venv\Scripts\activate"
Write-Host "     uv run python -m tiniestgpt.cli info"
Write-Host "     uv run python -m tiniestgpt.cli pretrain --config recipes/pretrain_tiny.yaml"
