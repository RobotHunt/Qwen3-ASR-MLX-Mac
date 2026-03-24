"""
Qwen3-ASR FastAPI 服务端 · OpenAI 兼容 API
专为 Apple Silicon (M1/M2/M3/M4) 设计
实时语音识别 · WebSocket 流式 · VAD 长音频分段

OpenAI 兼容端点:
  POST /v1/audio/transcriptions  — 对标 whisper-1
  GET  /v1/models                — 模型列表

原生端点:
  POST /transcribe               — 文件转写 (SSE/VAD)
  WS   /ws                       — 实时流式转写
  GET  /health                   — 健康检查
"""
import os
import time
import tempfile
import threading
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Generator, Callable

import numpy as np
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    WebSocket,
    WebSocketDisconnect,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel
from typing import List
import uvicorn

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps
except ImportError:
    torch = None
    load_silero_vad = None
    get_speech_timestamps = None


# ──────────────────────────────────────────
# 配置区：根据你的硬件修改
# ──────────────────────────────────────────
MODELS_DIR = os.path.expanduser("~/Downloads/Qwen3-ASR-Models")

# 模型注册表
MODEL_REGISTRY = {
    "qwen3-asr": {
        "path": f"{MODELS_DIR}/ASR-1.7B-8bit",
        "hf_fallback": "mlx-community/Qwen3-ASR-1.7B-8bit",
        "description": "Qwen3 语音识别模型 1.7B 参数 8-bit 量化，支持中英日韩等多语言",
        "capabilities": ["transcription", "streaming", "vad-segmentation"],
        "languages": ["zh", "en", "ja", "ko", "yue", "fr", "de", "es", "ru"],
    },
}

# 旧 ID 向后兼容映射
_LEGACY_MODEL_MAP = {"ASR-1.7B-8bit": "qwen3-asr"}

def resolve_model_path(model_id: str) -> str:
    """解析模型 ID 到实际路径，支持旧 ID 向后兼容"""
    # 兼容旧 ID
    if model_id in _LEGACY_MODEL_MAP:
        model_id = _LEGACY_MODEL_MAP[model_id]
    info = MODEL_REGISTRY.get(model_id)
    if not info:
        return None
    local_path = info["path"]
    if os.path.isdir(local_path):
        return local_path
    return info.get("hf_fallback", local_path)

# 默认模型
DEFAULT_MODEL_ID = "qwen3-asr"
MODEL_NAME = resolve_model_path(DEFAULT_MODEL_ID)

LANGUAGE = "zh"
SAMPLE_RATE = 16000

# VAD 参数
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_MS = 300
VAD_MIN_SPEECH_MS = 250

# OpenAI 兼容的响应格式
SUPPORTED_RESPONSE_FORMATS = ["json", "verbose_json", "text", "srt", "vtt"]


# ──────────────────────────────────────────
# 音频工具函数
# ──────────────────────────────────────────
def preprocess_audio(
    audio_input: str | np.ndarray,
    target_sr: int = 16000,
) -> tuple[np.ndarray, int]:
    """预处理音频到统一格式：16kHz mono float32"""
    if isinstance(audio_input, np.ndarray):
        return audio_input, target_sr

    audio_path = Path(audio_input)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_input}")

    if sf is None:
        raise RuntimeError("soundfile not installed")

    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        try:
            import resampy
            audio = resampy.resample(audio, sr, target_sr)
        except ImportError:
            pass
    return audio.astype(np.float32), target_sr


def save_temp_audio(audio: np.ndarray, sample_rate: int) -> str:
    """保存音频到临时 WAV 文件"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        if sf:
            sf.write(f.name, audio, sample_rate)
        else:
            import wave
            import struct
            with wave.open(f.name, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                audio_int16 = (audio * 32767).astype(np.int16)
                wf.writeframes(struct.pack("<" + "h" * len(audio_int16), *audio_int16))
        return f.name


# ──────────────────────────────────────────
# VAD 处理器
# ──────────────────────────────────────────
class VADProcessor:
    def __init__(
        self,
        threshold: float = VAD_THRESHOLD,
        min_silence_ms: int = VAD_MIN_SILENCE_MS,
        min_speech_ms: int = VAD_MIN_SPEECH_MS,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.sample_rate = sample_rate
        self._model = None
        self._audio_buffer = np.array([], dtype=np.float32)
        self._in_speech = False
        self._speech_start = 0
        self._silence_start = 0

    def load_model(self):
        if load_silero_vad is None:
            raise RuntimeError("silero-vad not installed")
        if self._model is None:
            self._model = load_silero_vad()
        return self._model

    def reset(self):
        self._audio_buffer = np.array([], dtype=np.float32)
        self._in_speech = False
        self._speech_start = 0
        self._silence_start = 0

    def process_chunk(self, audio: np.ndarray):
        """处理单个音频块，返回 (is_speech, speech_prob)"""
        if self._model is None:
            self.load_model()

        self._audio_buffer = np.concatenate([self._audio_buffer, audio])

        if len(self._audio_buffer) < 512:
            return False, 0.0

        chunk_tensor = torch.from_numpy(self._audio_buffer[-512:])
        with torch.no_grad():
            speech_prob = self._model(chunk_tensor, self.sample_rate).item()

        is_speech = speech_prob > self.threshold
        current_ms = len(self._audio_buffer) * 1000 // self.sample_rate

        if is_speech and not self._in_speech:
            self._in_speech = True
            self._speech_start = current_ms
        elif not is_speech and self._in_speech:
            if self._silence_start == 0:
                self._silence_start = current_ms
            elif current_ms - self._silence_start > self.min_silence_ms:
                self._in_speech = False
        elif is_speech:
            self._silence_start = 0

        return is_speech, speech_prob

    def get_speech_segments(self, audio: np.ndarray) -> list[dict]:
        if self._model is None:
            self.load_model()

        audio_tensor = torch.from_numpy(audio)
        with torch.no_grad():
            segments = get_speech_timestamps(
                audio_tensor,
                self._model,
                threshold=self.threshold,
                min_silence_duration_ms=self.min_silence_ms,
                min_speech_duration_ms=self.min_speech_ms,
            )
        return segments

    def split_by_silence(
        self, audio: np.ndarray, padding_samples: int = 800
    ) -> Generator[tuple[np.ndarray, int, int], None, None]:
        segments = self.get_speech_segments(audio)
        for seg in segments:
            start = max(0, seg["start"] - padding_samples)
            end = min(len(audio), seg["end"] + padding_samples)
            yield audio[start:end], start, end


# ──────────────────────────────────────────
# ASR 引擎
# ──────────────────────────────────────────
@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration: float
    rtf: float
    is_final: bool = True
    segment_id: int = 0


class ASREngine:
    def __init__(self, model_name: str = MODEL_NAME, language: str = LANGUAGE):
        self.model_name = model_name
        self.language = language
        self._model = None
        self._lock = threading.Lock()
        self._is_loaded = False

    def load_model(self):
        with self._lock:
            if self._is_loaded:
                return
            print(f"🔄 Loading ASR model: {self.model_name}")
            try:
                from mlx_audio.stt import load
                self._model = load(self.model_name)
                self._model._model.eval()
                print("✅ Model loaded successfully")
            except ImportError:
                print("❌ mlx-audio not available")
                self._model = None
            self._is_loaded = True

    def transcribe_file(
        self,
        audio_path: str,
        language: str = None,
        use_vad: bool = False,
    ) -> TranscriptionResult:
        if self._model is None:
            return TranscriptionResult(
                text="[Model not loaded]",
                language=language or self.language,
                duration=0, rtf=0,
            )

        start_time = time.time()

        try:
            audio, sr = preprocess_audio(audio_path, SAMPLE_RATE)
            temp_path = save_temp_audio(audio, sr)
            audio_duration = len(audio) / sr
            processed_path = temp_path
        except Exception as e:
            print(f"Audio preprocess failed, using original: {e}")
            processed_path = audio_path
            try:
                audio_data, sr = sf.read(audio_path)
                audio_duration = len(audio_data) / sr
            except:
                audio_duration = 0

        # VAD 分段批处理（长音频 >30s 优化）
        if use_vad and audio_duration > 30:
            result = self._transcribe_with_vad(processed_path, language, audio_duration)
            if processed_path != audio_path:
                Path(processed_path).unlink(missing_ok=True)
            return result

        result = self._model._model.generate(
            processed_path, language=language or self.language
        )

        if processed_path != audio_path:
            Path(processed_path).unlink(missing_ok=True)

        duration = time.time() - start_time
        rtf = duration / audio_duration if audio_duration > 0 else 0
        text = result.text if hasattr(result, "text") else str(result)

        return TranscriptionResult(
            text=text,
            language=language or self.language,
            duration=duration,
            rtf=rtf,
        )

    def _transcribe_with_vad(
        self, audio_path: str, language: str, audio_duration: float
    ) -> TranscriptionResult:
        """VAD 分段批处理转录"""
        start_time = time.time()
        audio, sr = preprocess_audio(audio_path, SAMPLE_RATE)

        vad = VADProcessor(sample_rate=sr)
        segments = list(vad.split_by_silence(audio))

        if not segments:
            result = self._model._model.generate(audio_path, language=language)
            return TranscriptionResult(
                text=result.text if hasattr(result, "text") else str(result),
                language=language,
                duration=time.time() - start_time,
                rtf=0,
            )

        print(f"📊 VAD detected {len(segments)} speech segments")

        results = []
        for seg_audio, seg_start, seg_end in segments:
            temp_path = save_temp_audio(seg_audio, sr)
            try:
                result = self._model._model.generate(temp_path, language=language)
                text = result.text if hasattr(result, "text") else str(result)
                results.append(text)
            finally:
                Path(temp_path).unlink(missing_ok=True)

        full_text = " ".join(r for r in results if r)
        duration = time.time() - start_time

        return TranscriptionResult(
            text=full_text,
            language=language,
            duration=duration,
            rtf=duration / audio_duration if audio_duration > 0 else 0,
        )

    def transcribe_audio(
        self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE, language: str = None
    ) -> TranscriptionResult:
        """转写 numpy 音频数组（WebSocket 用）"""
        if self._model is None:
            return TranscriptionResult(
                text="", language=language or self.language, duration=0, rtf=0
            )

        start_time = time.time()
        temp_path = save_temp_audio(audio, sample_rate)

        try:
            result = self._model._model.generate(
                temp_path, language=language or self.language
            )
        except Exception as e:
            print(f"Transcription error: {e}")
            result = None
        finally:
            Path(temp_path).unlink(missing_ok=True)

        duration = time.time() - start_time
        audio_duration = len(audio) / sample_rate
        rtf = duration / audio_duration if audio_duration > 0 else 0
        text = result.text if result and hasattr(result, "text") else ""

        return TranscriptionResult(
            text=text,
            language=language or self.language,
            duration=duration,
            rtf=rtf,
        )

    def stream_transcribe(
        self, audio_path: str, language: str = None
    ) -> Generator[TranscriptionResult, None, None]:
        """流式转写（SSE 用）"""
        if self._model is None:
            yield TranscriptionResult(
                text="[Model not loaded]",
                language=language or self.language,
                duration=0, rtf=0, is_final=False,
            )
            return

        segment_id = 0
        start_time = time.time()
        try:
            for chunk in self._model._model.stream_transcribe(
                audio_path, language=language or self.language
            ):
                text = chunk.text if hasattr(chunk, "text") else str(chunk)
                yield TranscriptionResult(
                    text=text,
                    language=language or self.language,
                    duration=time.time() - start_time,
                    rtf=0,
                    is_final=False,
                    segment_id=segment_id,
                )
                segment_id += 1
        except Exception as e:
            print(f"Streaming error: {e}")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self._model is not None


# ──────────────────────────────────────────
# Pydantic 响应模型
# ──────────────────────────────────────────
class TranscriptionResponse(BaseModel):
    text: str
    language: str
    duration: float
    rtf: float
    timestamp: datetime


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str


# ──────────────────────────────────────────
# FastAPI 应用
# ──────────────────────────────────────────
PORT = 8001  # TTS 服务占用 8000，ASR 使用 8001

asr_engine = ASREngine()
start_time = time.time()


@asynccontextmanager
async def lifespan(app):
    asr_engine.load_model()
    print(f"🚀 Qwen3-ASR Server Started | Model: {MODEL_NAME} | Port: {PORT}")
    yield
    print("Shutting down ASR Server...")


app = FastAPI(
    title="Qwen3-ASR Apple Silicon API",
    version="1.0",
    description="本地离线语音识别服务 · 实时流式转写 · VAD 长音频分段",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "name": "Qwen3-ASR Apple Silicon",
        "version": "1.0",
        "model": MODEL_NAME,
        "uptime": time.time() - start_time,
    }


def cleanup_file(path: str):
    Path(path).unlink(missing_ok=True)


# ──────────────────────────────────────────
# OpenAI 兼容端点
# ──────────────────────────────────────────

@app.get("/v1/models")
async def openai_list_models():
    """
    OpenAI 兼容：列出可用模型
    对标 GET /v1/models
    """
    now = int(time.time())
    data = []
    for model_id, info in MODEL_REGISTRY.items():
        model_path = resolve_model_path(model_id)
        data.append({
            "id": model_id,
            "object": "model",
            "created": now,
            "owned_by": "qwen3-asr-mlx",
            "description": info["description"],
            "capabilities": info["capabilities"],
            "languages": info["languages"],
            "ready": os.path.isdir(info["path"]),
        })
    return {"object": "list", "data": data}


def _format_srt(text: str, duration: float) -> str:
    """生成 SRT 字幕格式"""
    def _ts(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    return f"1\n{_ts(0.0)} --> {_ts(duration)}\n{text}\n"


def _format_vtt(text: str, duration: float) -> str:
    """生成 WebVTT 字幕格式"""
    def _ts(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    return f"WEBVTT\n\n{_ts(0.0)} --> {_ts(duration)}\n{text}\n"


def _format_srt_segments(segments: list) -> str:
    """从 VAD 分段结果生成 SRT"""
    def _ts(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(f"{i}")
        lines.append(f"{_ts(seg['start'])} --> {_ts(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def _format_vtt_segments(segments: list) -> str:
    """从 VAD 分段结果生成 VTT"""
    def _ts(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_ts(seg['start'])} --> {_ts(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


@app.post("/v1/audio/transcriptions")
async def openai_transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file (wav/mp3/m4a/flac/ogg)"),
    model: str = Form(default="qwen3-asr", description="Model ID"),
    language: str = Form(default="zh", description="ISO language code: zh, en, ja, ko, etc."),
    response_format: str = Form(default="json", description="Response format: json, verbose_json, text, srt, vtt"),
    temperature: float = Form(default=0, description="Reserved for compatibility"),
):
    """
    OpenAI 兼容：音频转写
    对标 POST /v1/audio/transcriptions (whisper-1)

    支持的 response_format:
      - json: {"text": "..."}
      - verbose_json: {"task": "transcribe", "language": "zh", "duration": 3.5, "text": "...", "segments": [...]}
      - text: 纯文本
      - srt: SRT 字幕格式
      - vtt: WebVTT 字幕格式
    """
    # ── 1. 校验参数 ──
    # 兼容旧 ID
    if model in _LEGACY_MODEL_MAP:
        model = _LEGACY_MODEL_MAP[model]

    if model not in MODEL_REGISTRY:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format '{response_format}'. Supported: {SUPPORTED_RESPONSE_FORMATS}"
        )

    # ── 2. 保存上传文件 ──
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(file.filename).suffix if file.filename else ".wav"
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    background_tasks.add_task(cleanup_file, tmp_path)

    # ── 3. 获取音频时长 ──
    audio_duration = 0.0
    try:
        audio_data, sr = sf.read(tmp_path)
        audio_duration = len(audio_data) / sr
    except:
        pass

    # ── 4. 决定是否使用 VAD 分段 (长音频自动启用) ──
    use_vad = audio_duration > 30

    # ── 5. 转写 ──
    asr_engine.load_model()

    if use_vad and (response_format in ["verbose_json", "srt", "vtt"]):
        # VAD 分段转写，保留每段的时间戳
        segments = _transcribe_with_segments(tmp_path, language, audio_duration)
        full_text = " ".join(s["text"] for s in segments if s["text"])

        if response_format == "verbose_json":
            return JSONResponse({
                "task": "transcribe",
                "language": language,
                "duration": round(audio_duration, 2),
                "text": full_text,
                "segments": [
                    {
                        "id": i,
                        "start": round(s["start"], 2),
                        "end": round(s["end"], 2),
                        "text": s["text"],
                    }
                    for i, s in enumerate(segments)
                ],
            })
        elif response_format == "srt":
            return PlainTextResponse(_format_srt_segments(segments), media_type="text/plain")
        elif response_format == "vtt":
            return PlainTextResponse(_format_vtt_segments(segments), media_type="text/plain")
    else:
        result = asr_engine.transcribe_file(tmp_path, language, use_vad=use_vad)
        full_text = result.text

    # ── 6. 格式化响应 ──
    if response_format == "json":
        return JSONResponse({"text": full_text})

    elif response_format == "verbose_json":
        return JSONResponse({
            "task": "transcribe",
            "language": language,
            "duration": round(audio_duration, 2),
            "text": full_text,
            "segments": [
                {"id": 0, "start": 0.0, "end": round(audio_duration, 2), "text": full_text}
            ],
        })

    elif response_format == "text":
        return PlainTextResponse(full_text)

    elif response_format == "srt":
        return PlainTextResponse(_format_srt(full_text, audio_duration), media_type="text/plain")

    elif response_format == "vtt":
        return PlainTextResponse(_format_vtt(full_text, audio_duration), media_type="text/plain")


def _transcribe_with_segments(
    audio_path: str, language: str, audio_duration: float
) -> list:
    """
    VAD 分段转写，返回带时间戳的 segments 列表
    [{"text": "...", "start": 0.0, "end": 3.5}, ...]
    """
    audio, sr = preprocess_audio(audio_path, SAMPLE_RATE)
    vad = VADProcessor(sample_rate=sr)
    vad_segments = list(vad.split_by_silence(audio))

    if not vad_segments:
        result = asr_engine.transcribe_file(audio_path, language)
        return [{"text": result.text, "start": 0.0, "end": audio_duration}]

    results = []
    for seg_audio, seg_start, seg_end in vad_segments:
        temp_path = save_temp_audio(seg_audio, sr)
        try:
            result = asr_engine._model._model.generate(temp_path, language=language)
            text = result.text if hasattr(result, "text") else str(result)
            results.append({
                "text": text,
                "start": seg_start / sr,
                "end": seg_end / sr,
            })
        finally:
            Path(temp_path).unlink(missing_ok=True)

    return results


if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=PORT, reload=False)
