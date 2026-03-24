#!/bin/bash
set -e

echo "🚀 Qwen3-ASR Apple Silicon 一键安装"
echo "======================================"

# 1. 创建虚拟环境
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
    echo "📦 创建 Python 虚拟环境..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# 2. 安装依赖
echo "📦 安装依赖..."
pip install -q -r requirements.txt

# 3. 下载模型
echo "⬇️  下载 ASR 框架和缓存模型架构..."
python -c "
import os
try:
    from mlx_audio.stt import load
    print('Loading ASR model to cache...')
    load('mlx-community/Qwen3-ASR-1.7B-8bit')
    print('✅ ASR 模型就绪 (通过 mlx-audio 自动管理缓存)')
except ImportError:
    print('⚠️ 需要网络连接，首次运行时将自动下载模型')
"

echo ""
echo "🎉 安装完成！"
echo ""
echo "使用方式："
echo "  source .venv/bin/activate"
echo "  python demo.py transcribe test.wav     # 基础转写"
echo "  python demo.py stream test.wav         # 流式转写"
echo "  python demo.py vad test.wav            # VAD 分段长音频"
echo "  python demo.py benchmark test.wav      # RTF 性能测试"
echo "  python fastapi_server.py               # 启动 API 服务器"
