"""
Qwen3-ASR FastAPI 服务端 · OpenAI 兼容 API
专为 Apple Silicon (M1/M2/M3/M4) 设计
实时语音识别 · VAD 长音频分段

OpenAI 兼容端点:
  POST /v1/audio/transcriptions         — 对标 whisper-1
  WS   /v1/audio/transcriptions/stream  — 实时流式转写
  GET  /v1/models                       — 模型列表
"""
import os
import time
import tempfile
import threading
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Generator

import numpy as np
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    WebSocket,
    WebSocketDisconnect,
    BackgroundTasks,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse, HTMLResponse
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
# Playground HTML
# ──────────────────────────────────────────
PLAYGROUND_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ASR Playground</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family:'Inter',system-ui,-apple-system,sans-serif;
    background:#0a0a0f;
    color:#e4e4e7;
    min-height:100vh;
    display:flex; align-items:center; justify-content:center;
    position:relative; overflow:hidden;
  }
  body::before {
    content:''; position:absolute; top:-50%; left:-50%;
    width:200%; height:200%;
    background:radial-gradient(circle at 30% 40%, rgba(59,130,246,0.08) 0%, transparent 50%),
               radial-gradient(circle at 70% 60%, rgba(168,85,247,0.06) 0%, transparent 50%);
    animation:bgShift 20s ease-in-out infinite;
  }
  @keyframes bgShift { 0%,100%{transform:translate(0,0)} 50%{transform:translate(-3%,2%)} }
  .container {
    position:relative; z-index:1;
    width:100%; max-width:640px; padding:20px;
  }
  .card {
    background:rgba(24,24,32,0.85);
    backdrop-filter:blur(20px);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:20px;
    padding:36px 32px;
    box-shadow:0 20px 60px rgba(0,0,0,0.5);
  }
  .title {
    font-size:22px; font-weight:700;
    background:linear-gradient(135deg,#60a5fa,#a78bfa,#f472b6);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    text-align:center; margin-bottom:4px;
  }
  .subtitle {
    text-align:center; color:#71717a; font-size:13px; margin-bottom:28px;
  }
  .controls {
    display:grid; grid-template-columns:1fr 1fr; gap:12px;
    margin-bottom:20px;
  }
  label { font-size:12px; color:#a1a1aa; margin-bottom:4px; display:block; font-weight:500; }
  select, input[type=file] {
    width:100%; padding:10px 12px;
    background:rgba(39,39,42,0.8);
    border:1px solid rgba(255,255,255,0.1);
    border-radius:10px; color:#e4e4e7;
    font-size:13px; outline:none;
    transition:border-color 0.2s;
  }
  select:focus { border-color:#60a5fa; }
  .file-zone {
    border:2px dashed rgba(255,255,255,0.1);
    border-radius:12px; padding:20px;
    text-align:center; color:#71717a;
    font-size:13px; cursor:pointer;
    transition:all 0.2s; margin-bottom:20px;
    position:relative;
  }
  .file-zone:hover, .file-zone.dragover {
    border-color:#60a5fa; color:#60a5fa;
    background:rgba(59,130,246,0.05);
  }
  .file-zone input { position:absolute; inset:0; opacity:0; cursor:pointer; }
  .file-zone .name { color:#a78bfa; font-weight:500; margin-top:6px; }
  .rec-row { display:flex; gap:12px; margin-bottom:24px; }
  .btn {
    flex:1; padding:14px;
    border:none; border-radius:12px;
    font-size:14px; font-weight:600;
    cursor:pointer; transition:all 0.25s;
    display:flex; align-items:center; justify-content:center; gap:8px;
  }
  .btn-rec {
    background:linear-gradient(135deg,#ef4444,#dc2626);
    color:#fff;
  }
  .btn-rec:hover { transform:translateY(-1px); box-shadow:0 6px 20px rgba(239,68,68,0.3); }
  .btn-rec.recording {
    background:linear-gradient(135deg,#f97316,#ea580c);
    animation:pulse 1.5s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(249,115,22,0.4)} 50%{box-shadow:0 0 0 12px rgba(249,115,22,0)} }
  .btn-send {
    background:linear-gradient(135deg,#3b82f6,#6366f1);
    color:#fff;
  }
  .btn-send:hover { transform:translateY(-1px); box-shadow:0 6px 20px rgba(59,130,246,0.3); }
  .btn:disabled { opacity:0.4; cursor:not-allowed; transform:none!important; box-shadow:none!important; }
  canvas#waveform {
    width:100%; height:64px;
    border-radius:10px;
    background:rgba(39,39,42,0.5);
    margin-bottom:20px; display:none;
  }
  .result-box {
    background:rgba(39,39,42,0.6);
    border:1px solid rgba(255,255,255,0.06);
    border-radius:12px; padding:20px;
    display:none; animation:fadeIn 0.3s;
  }
  @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:none} }
  .result-box h3 {
    font-size:13px; color:#a78bfa; font-weight:600;
    margin-bottom:10px; display:flex; align-items:center; gap:6px;
  }
  .result-text {
    font-size:15px; line-height:1.7; color:#f4f4f5;
    white-space:pre-wrap; word-break:break-word;
  }
  .result-meta {
    margin-top:12px; padding-top:12px;
    border-top:1px solid rgba(255,255,255,0.06);
    font-size:12px; color:#71717a;
    display:flex; gap:16px; flex-wrap:wrap;
  }
  .spinner {
    width:18px; height:18px;
    border:2px solid rgba(255,255,255,0.2);
    border-top-color:#60a5fa;
    border-radius:50%;
    animation:spin 0.6s linear infinite;
    display:inline-block;
  }
  @keyframes spin { to{transform:rotate(360deg)} }
  .status { text-align:center; font-size:13px; color:#71717a; margin-bottom:16px; min-height:20px; }
</style>
</head>
<body>
<div class="container">
<div class="card">
  <div class="title">🎙 ASR Playground</div>
  <div class="subtitle">OpenAI-Compatible Speech Recognition · Apple Silicon</div>

  <div class="controls">
    <div>
      <label>语言 Language</label>
      <select id="lang">
        <option value="zh" selected>中文 Chinese</option>
        <option value="en">English</option>
        <option value="ja">日本語 Japanese</option>
        <option value="ko">한국어 Korean</option>
        <option value="yue">粤语 Cantonese</option>
        <option value="fr">Français French</option>
        <option value="de">Deutsch German</option>
        <option value="es">Español Spanish</option>
        <option value="ru">Русский Russian</option>
      </select>
    </div>
    <div>
      <label>输出格式 Format</label>
      <select id="fmt">
        <option value="json" selected>JSON</option>
        <option value="verbose_json">Verbose JSON</option>
        <option value="text">Text</option>
        <option value="srt">SRT Subtitle</option>
        <option value="vtt">WebVTT Subtitle</option>
      </select>
    </div>
  </div>

  <div class="file-zone" id="dropZone">
    <div>📁 拖放音频文件到此处，或点击选择</div>
    <div>支持 WAV / MP3 / M4A / FLAC / OGG</div>
    <div class="name" id="fileName"></div>
    <input type="file" id="fileInput" accept="audio/*">
  </div>

  <canvas id="waveform"></canvas>

  <div class="rec-row">
    <button class="btn btn-rec" id="recBtn" onclick="toggleRecord()">
      <span id="recIcon">⏺</span> <span id="recLabel">录音</span>
    </button>
    <button class="btn btn-send" id="sendBtn" onclick="sendAudio()" disabled>
      ▶ 识别
    </button>
  </div>

  <div class="status" id="status"></div>
  <div class="result-box" id="resultBox">
    <h3><span>✨</span> 识别结果</h3>
    <div class="result-text" id="resultText"></div>
    <div class="result-meta" id="resultMeta"></div>
  </div>
</div>
</div>

<script>
let mediaRecorder, audioChunks=[], audioBlob=null, isRecording=false;
let analyser, animId, canvasCtx;
const canvas = document.getElementById('waveform');

// ── File handling ──
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('dragover');
  if(e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', e => { if(e.target.files.length) handleFile(e.target.files[0]); });

function handleFile(file) {
  audioBlob = file;
  document.getElementById('fileName').textContent = file.name;
  document.getElementById('sendBtn').disabled = false;
  setStatus('📄 已选择: ' + file.name);
}

// ── Recording ──
async function toggleRecord() {
  if(isRecording) { stopRecord(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio:{sampleRate:16000,channelCount:1}});
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream, {mimeType:'audio/webm;codecs=opus'});
    mediaRecorder.ondataavailable = e => { if(e.data.size>0) audioChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach(t=>t.stop());
      audioBlob = new Blob(audioChunks, {type:'audio/webm'});
      document.getElementById('sendBtn').disabled = false;
      document.getElementById('fileName').textContent = '';
      setStatus('🎤 录音完成 (' + (audioChunks.length) + ' chunks)');
      stopWaveform();
    };
    mediaRecorder.start(250);
    isRecording = true;
    document.getElementById('recBtn').classList.add('recording');
    document.getElementById('recIcon').textContent = '⏹';
    document.getElementById('recLabel').textContent = '停止';
    setStatus('🔴 录音中...');

    // Waveform
    const actx = new AudioContext();
    const src = actx.createMediaStreamSource(stream);
    analyser = actx.createAnalyser();
    analyser.fftSize = 256;
    src.connect(analyser);
    canvas.style.display = 'block';
    canvasCtx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = canvas.offsetHeight * 2;
    drawWaveform();
  } catch(err) {
    setStatus('❌ 麦克风权限被拒绝');
  }
}

function stopRecord() {
  if(mediaRecorder && mediaRecorder.state!=='inactive') mediaRecorder.stop();
  isRecording = false;
  document.getElementById('recBtn').classList.remove('recording');
  document.getElementById('recIcon').textContent = '⏺';
  document.getElementById('recLabel').textContent = '录音';
}

function drawWaveform() {
  if(!analyser) return;
  const bufLen = analyser.frequencyBinCount;
  const data = new Uint8Array(bufLen);
  const w = canvas.width, h = canvas.height;
  function draw() {
    animId = requestAnimationFrame(draw);
    analyser.getByteTimeDomainData(data);
    canvasCtx.fillStyle = 'rgba(39,39,42,0.3)';
    canvasCtx.fillRect(0,0,w,h);
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = '#60a5fa';
    canvasCtx.beginPath();
    const sliceW = w/bufLen;
    let x = 0;
    for(let i=0;i<bufLen;i++){
      const v = data[i]/128.0;
      const y = v*h/2;
      i===0 ? canvasCtx.moveTo(x,y) : canvasCtx.lineTo(x,y);
      x+=sliceW;
    }
    canvasCtx.lineTo(w,h/2);
    canvasCtx.stroke();
  }
  draw();
}

function stopWaveform() {
  if(animId) cancelAnimationFrame(animId);
  setTimeout(()=>{ canvas.style.display='none'; }, 1000);
}

// ── Send ──
async function sendAudio() {
  if(!audioBlob) return;
  const lang = document.getElementById('lang').value;
  const fmt = document.getElementById('fmt').value;
  const sendBtn = document.getElementById('sendBtn');
  const resultBox = document.getElementById('resultBox');
  const resultText = document.getElementById('resultText');
  const resultMeta = document.getElementById('resultMeta');

  sendBtn.disabled = true;
  sendBtn.innerHTML = '<span class="spinner"></span> 识别中...';
  resultBox.style.display = 'none';
  setStatus('⏳ 正在转写...');

  const fd = new FormData();
  const ext = audioBlob.type.includes('webm') ? 'webm' : (audioBlob.name||'audio').split('.').pop();
  fd.append('file', audioBlob, 'recording.' + ext);
  fd.append('model', 'qwen3-asr');
  fd.append('language', lang);
  fd.append('response_format', fmt);

  const t0 = performance.now();
  try {
    const resp = await fetch('/v1/audio/transcriptions', {method:'POST', body:fd});
    const elapsed = ((performance.now()-t0)/1000).toFixed(2);

    if(!resp.ok) {
      const err = await resp.text();
      resultText.textContent = '❌ Error: ' + err;
      resultMeta.innerHTML = '';
      resultBox.style.display = 'block';
      setStatus('');
      return;
    }

    if(fmt==='json') {
      const j = await resp.json();
      resultText.textContent = j.text;
      resultMeta.innerHTML = '<span>⏱ ' + elapsed + 's</span><span>📋 JSON</span>';
    } else if(fmt==='verbose_json') {
      const j = await resp.json();
      resultText.textContent = j.text;
      let meta = '<span>⏱ ' + elapsed + 's</span>'
        + '<span>🕐 ' + j.duration + 's</span>'
        + '<span>📊 ' + (j.segments||[]).length + ' segments</span>';
      resultMeta.innerHTML = meta;
    } else {
      const text = await resp.text();
      resultText.textContent = text;
      resultMeta.innerHTML = '<span>⏱ ' + elapsed + 's</span><span>📋 ' + fmt.toUpperCase() + '</span>';
    }
    resultBox.style.display = 'block';
    setStatus('✅ 完成');
  } catch(e) {
    resultText.textContent = '❌ ' + e.message;
    resultMeta.innerHTML = '';
    resultBox.style.display = 'block';
    setStatus('');
  } finally {
    sendBtn.disabled = false;
    sendBtn.innerHTML = '▶ 识别';
  }
}

function setStatus(s) { document.getElementById('status').textContent = s; }
</script>
</body>
</html>"""

PORT = 8088

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


@app.get("/", response_class=HTMLResponse)
async def playground():
    """ASR Playground — 浏览器录音测试页"""
    return HTMLResponse(PLAYGROUND_HTML)


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

    # ── 2.5 转换不支持的格式 (webm/ogg/opus → wav) ──
    ext = Path(tmp_path).suffix.lower()
    if ext in [".webm", ".ogg", ".opus", ".weba"]:
        wav_path = tmp_path.rsplit(".", 1)[0] + ".wav"
        try:
            import subprocess, shutil
            ffmpeg_bin = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
            result = subprocess.run(
                [ffmpeg_bin, "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
                capture_output=True, timeout=30,
            )
            if result.returncode == 0 and Path(wav_path).exists():
                Path(tmp_path).unlink(missing_ok=True)
                tmp_path = wav_path
            else:
                print(f"⚠️ ffmpeg 转换失败 (code={result.returncode}): {result.stderr.decode()[:200]}")
        except Exception as e:
            print(f"⚠️ ffmpeg 不可用: {e}")

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


@app.websocket("/v1/audio/transcriptions/stream")
async def openai_ws_transcribe(websocket: WebSocket):
    """
    OpenAI 扩展：WebSocket 实时流式转写

    客户端发送: float32 音频块 (16kHz mono)
    服务端返回: JSON {"text": "...", "segment_id": 0, "is_final": true}
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

            # 语音结束且缓冲区足够长时触发转写
            if not is_speech and len(audio_buffer) > SAMPLE_RATE:
                if len(audio_buffer) > SAMPLE_RATE * 0.5:
                    result = asr_engine.transcribe_audio(audio_buffer)
                    await websocket.send_json({
                        "text": result.text,
                        "segment_id": segment_id,
                        "is_final": True,
                        "duration": result.duration,
                        "timestamp": datetime.now().isoformat(),
                    })
                    segment_id += 1

                audio_buffer = np.array([], dtype=np.float32)
                vad.reset()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass


if __name__ == "__main__":
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=PORT, reload=False)
