"""
Qwen3-ASR FastAPI 服务端
专为 Apple Silicon (M1/M2/M3/M4) 设计
实时语音识别 · WebSocket 流式 · VAD 长音频分段
"""
import os
import time
import tempfile
import threading
from pathlib import Path
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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
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
MODEL_MAP = {
    "ASR-1.7B-8bit": f"{MODELS_DIR}/ASR-1.7B-8bit",
}

# 优先使用本地模型，不存在则回退到 HuggingFace 名称
_default_model_key = "ASR-1.7B-8bit"
_local_path = MODEL_MAP[_default_model_key]
MODEL_NAME = _local_path if os.path.isdir(_local_path) else "mlx-community/Qwen3-ASR-1.7B-8bit"

LANGUAGE = "zh"
SAMPLE_RATE = 16000

# VAD 参数
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_MS = 300
VAD_MIN_SPEECH_MS = 250


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
asr_engine = ASREngine()
start_time = time.time()

app = FastAPI(
    title="Qwen3-ASR Apple Silicon API",
    version="1.0",
    description="本地离线语音识别服务 · 实时流式转写 · VAD 长音频分段",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    asr_engine.load_model()
    print(f"🚀 Qwen3-ASR Server Started | Model: {MODEL_NAME}")


@app.get("/")
async def root():
    return {
        "name": "Qwen3-ASR Apple Silicon",
        "version": "1.0",
        "model": MODEL_NAME,
        "uptime": time.time() - start_time,
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=asr_engine.is_loaded,
        model_name=MODEL_NAME,
    )


def cleanup_file(path: str):
    Path(path).unlink(missing_ok=True)


@app.post("/transcribe")
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="音频文件"),
    language: str = Form(default="zh", description="语言: zh, en, ja, ko 等"),
    stream: bool = Form(default=False, description="SSE 流式输出"),
    use_vad: bool = Form(default=False, description="VAD 分段（长音频 >30s）"),
):
    """
    转写音频文件

    - stream=false: 返回完整结果
    - stream=true: SSE 流式返回
    - use_vad=true: 长音频自动 VAD 分段
    """
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(file.filename).suffix if file.filename else ".wav"
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    background_tasks.add_task(cleanup_file, tmp_path)

    if stream:
        def generate():
            for chunk in asr_engine.stream_transcribe(tmp_path, language):
                yield f"data: {chunk.text}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            background=background_tasks,
        )

    result = asr_engine.transcribe_file(tmp_path, language, use_vad=use_vad)

    return TranscriptionResponse(
        text=result.text,
        language=result.language,
        duration=result.duration,
        rtf=result.rtf,
        timestamp=datetime.now(),
    )


@app.websocket("/ws")
async def websocket_transcribe(websocket: WebSocket):
    """
    WebSocket 实时转写

    发送: float32 音频块 (16kHz mono)
    接收: JSON {"text": "...", "segment_id": 0, "rtf": 0.1}
    """
    await websocket.accept()
    asr_engine.load_model()

    vad = VADProcessor()
    audio_buffer = np.array([], dtype=np.float32)
    segment_id = 0

    try:
        while True:
            data = await websocket.receive_bytes()
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            audio_buffer = np.concatenate([audio_buffer, audio_chunk])

            is_speech, _ = vad.process_chunk(audio_chunk)

            if not is_speech and len(audio_buffer) > SAMPLE_RATE:
                if len(audio_buffer) > SAMPLE_RATE * 0.5:
                    result = asr_engine.transcribe_audio(audio_buffer)
                    await websocket.send_json({
                        "text": result.text,
                        "segment_id": segment_id,
                        "rtf": result.rtf,
                        "timestamp": datetime.now().isoformat(),
                    })
                    segment_id += 1

                audio_buffer = np.array([], dtype=np.float32)
                vad.reset()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"error": str(e)})


if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8000, reload=False)
