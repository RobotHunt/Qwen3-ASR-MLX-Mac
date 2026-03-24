# Qwen3-ASR Apple Silicon

> 🚀 完全离线的高性能本地语音识别服务，专为 Mac M 系列芯片深度优化
> 基于 [MLX](https://github.com/ml-explore/mlx) 框架 + Metal GPU 加速
> **OpenAI API 兼容** — 直接对接支持 Whisper 的客户端

---

## 📁 项目结构

```
Qwen3-ASR-MLX-Mac/
├── README.md              # 本文档
├── setup.sh               # 一键安装 + 模型缓存
├── requirements.txt       # Python 核心依赖
├── demo.py                # CLI 综合演示工具
└── fastapi_server.py      # FastAPI 服务端 (OpenAI 兼容)
```

---

## ⚡ 快速开始

```bash
# 一键安装
chmod +x setup.sh && ./setup.sh

# 激活环境
source .venv/bin/activate

# 启动 API 服务器 (端口 8001)
python fastapi_server.py
```

---

## 🚀 API 接口

### `GET /v1/models` — 模型列表

```bash
curl -s http://localhost:8001/v1/models | python3 -m json.tool
```

返回模型能力、支持语言、就绪状态。

### `POST /v1/audio/transcriptions` — 语音转写

对标 OpenAI `whisper-1` 接口，支持 5 种响应格式。

**基础转写 (json)**

```bash
curl -X POST "http://localhost:8001/v1/audio/transcriptions" \
  -F "file=@test.wav" \
  -F "model=qwen3-asr" \
  -F "language=zh"
# {"text": "你好世界"}
```

**详细 JSON (verbose_json)**

```bash
curl -X POST "http://localhost:8001/v1/audio/transcriptions" \
  -F "file=@test.wav" \
  -F "response_format=verbose_json"
# {"task":"transcribe","language":"zh","duration":3.0,"text":"...","segments":[...]}
```

**纯文本 / SRT 字幕 / WebVTT 字幕**

```bash
# 纯文本
curl -X POST "http://localhost:8001/v1/audio/transcriptions" \
  -F "file=@test.wav" -F "response_format=text"

# SRT 字幕
curl -X POST "http://localhost:8001/v1/audio/transcriptions" \
  -F "file=@test.wav" -F "response_format=srt"

# WebVTT 字幕
curl -X POST "http://localhost:8001/v1/audio/transcriptions" \
  -F "file=@test.wav" -F "response_format=vtt"
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `file` | file | 必需 | 音频文件 (wav/mp3/m4a/flac/ogg) |
| `model` | string | `qwen3-asr` | 模型 ID |
| `language` | string | `zh` | 语言: zh, en, ja, ko 等 |
| `response_format` | string | `json` | json / verbose_json / text / srt / vtt |
| `temperature` | float | `0` | 保留字段（兼容用） |

> 长音频 (>30s) 自动启用 VAD 分段，`verbose_json`/`srt`/`vtt` 格式下保留每段时间戳。

---

## 📝 许可

模型权重来自 [mlx-community/Qwen3-ASR-1.7B-8bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit)，请遵循其许可协议。

*Built for Apple Silicon · Powered by MLX · 2026.03*
