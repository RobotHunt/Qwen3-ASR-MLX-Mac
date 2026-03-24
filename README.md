# Qwen3-ASR Apple Silicon

> 🚀 完全离线的高性能本地语音识别服务，专为 Mac M 系列芯片深度优化
> 基于 [MLX](https://github.com/ml-explore/mlx) 框架 + Metal GPU 加速

---

## 📁 项目结构

```
Qwen3-ASR-MLX-Mac/
├── README.md              # 本文档
├── setup.sh               # 一键安装 + 模型缓存
├── requirements.txt       # Python 核心依赖
├── demo.py                # CLI 综合演示工具
└── fastapi_server.py      # FastAPI 生产级服务端
```

---

## ⚡ 快速开始

```bash
# 一键安装 (创建环境 + 安装依赖 + 缓存 8-bit 模型)
chmod +x setup.sh && ./setup.sh

# 激活环境
source .venv/bin/activate

# 运行演示
python demo.py transcribe test.wav     # 基础转写
python demo.py stream test.wav         # 流式转写
python demo.py vad test.wav            # VAD 分段长音频
python demo.py benchmark test.wav      # RTF 性能压测

# 启动 API 服务器
python fastapi_server.py
```

---

## 🚀 API 服务端

### 架构特性

1. **统一的 FastAPI 服务**：将转写、流式、VAD 逻辑完美合并在单文件内
2. **WebSocket 实时流式**：向服务端推音频流，即时拿到增量识别文本
3. **SSE 支持**：支持在 HTTP POST 时使用 `stream=true` 获取 Server-Sent Events

### 启动

```bash
python fastapi_server.py
# 访问 Swagger UI: http://localhost:8000/docs
```

### API 接口使用

**1. POST /transcribe — 基础 / 长音频 VAD 转写**

```bash
# 基础转写
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@test.wav" \
  -F "language=zh"

# 长音频自动 VAD 分段
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@long_audio.wav" \
  -F "use_vad=true"
```

**2. POST /transcribe — SSE 流式转写**

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@test.wav" \
  -F "stream=true"
```

**3. WebSocket /ws — 实时双向流转写**

```python
import websockets
import asyncio
import numpy as np

async def live_transcribe():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        # 发送 float32 音频块 (16kHz mono)
        chunk = np.random.randn(480).astype(np.float32)
        await ws.send(chunk.tobytes())
        
        # 接收 JSON: {"text": "...", "segment_id": 0, "rtf": 0.1, "timestamp": "..."}
        result = await ws.recv()
        print(result)

asyncio.run(live_transcribe())
```

---

## 📝 许可

本项目提供精简版的部署脚本和服务端代码。模型权重来自 [mlx-community/Qwen3-ASR-1.7B-8bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit)，请遵循其许可协议。

*Built for Apple Silicon · Powered by MLX · 2026.03*
