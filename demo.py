"""
Qwen3-ASR Apple Silicon 综合演示脚本
支持模式：文件转写 / VAD 分段 / 流式转写 / 性能测试
"""
import os
import sys
import time
import argparse

try:
    from mlx_audio.stt import load
except ImportError:
    print("❌ 请先安装依赖: pip install mlx mlx-audio soundfile")
    sys.exit(1)

import soundfile as sf
import numpy as np

# 模型配置：优先使用本地下载的模型
MODELS_DIR = os.path.expanduser("~/Downloads/Qwen3-ASR-Models")
_local_path = os.path.join(MODELS_DIR, "ASR-1.7B-8bit")
MODEL_NAME = _local_path if os.path.isdir(_local_path) else "mlx-community/Qwen3-ASR-1.7B-8bit"


def get_model(model_name: str = MODEL_NAME):
    """加载 ASR 模型（带缓存）"""
    if not hasattr(get_model, "_cache"):
        get_model._cache = {}
    if model_name not in get_model._cache:
        print(f"🔄 Loading model: {model_name}")
        model = load(model_name)
        model._model.eval()
        get_model._cache[model_name] = model
        print("✅ Model loaded")
    return get_model._cache[model_name]


def demo_transcribe(args):
    """基础文件转写"""
    print(f"\n🔹 转写文件: {args.audio}")
    model = get_model()
    start = time.time()
    result = model._model.generate(args.audio, language=args.language)
    elapsed = time.time() - start

    audio_data, sr = sf.read(args.audio)
    audio_duration = len(audio_data) / sr
    rtf = elapsed / audio_duration if audio_duration > 0 else 0

    text = result.text if hasattr(result, "text") else str(result)
    print(f"📝 结果: {text}")
    print(f"⏱️  耗时: {elapsed:.2f}s | 音频时长: {audio_duration:.2f}s | RTF: {rtf:.4f}")


def demo_vad(args):
    """VAD 分段转写（长音频优化）"""
    print(f"\n🔹 VAD 分段转写: {args.audio}")

    try:
        import torch
        from silero_vad import load_silero_vad, get_speech_timestamps
    except ImportError:
        print("❌ 需要 silero-vad: pip install silero-vad torch")
        return

    audio_data, sr = sf.read(args.audio)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    audio_data = audio_data.astype(np.float32)
    audio_duration = len(audio_data) / sr
    print(f"📊 音频时长: {audio_duration:.2f}s")

    # VAD 分段
    vad_model = load_silero_vad()
    audio_tensor = torch.from_numpy(audio_data)
    segments = get_speech_timestamps(
        audio_tensor, vad_model,
        threshold=0.5,
        min_silence_duration_ms=300,
        min_speech_duration_ms=250,
    )
    print(f"📊 检测到 {len(segments)} 个语音段")

    model = get_model()
    start = time.time()
    results = []

    for i, seg in enumerate(segments):
        seg_start = max(0, seg["start"] - 800)
        seg_end = min(len(audio_data), seg["end"] + 800)
        seg_audio = audio_data[seg_start:seg_end]

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, seg_audio, sr)
            temp_path = f.name

        try:
            result = model._model.generate(temp_path, language=args.language)
            text = result.text if hasattr(result, "text") else str(result)
            results.append(text)
            seg_time = seg_start / sr
            print(f"  [{seg_time:.1f}s] {text}")
        finally:
            os.unlink(temp_path)

    elapsed = time.time() - start
    full_text = " ".join(r for r in results if r)
    print(f"\n📝 完整结果: {full_text}")
    print(f"⏱️  耗时: {elapsed:.2f}s | RTF: {elapsed / audio_duration:.4f}")


def demo_stream(args):
    """流式转写"""
    print(f"\n🔹 流式转写: {args.audio}")
    model = get_model()
    start = time.time()

    print("📝 流式输出:")
    segment_id = 0
    for chunk in model._model.stream_transcribe(args.audio, language=args.language):
        text = chunk.text if hasattr(chunk, "text") else str(chunk)
        print(f"  [seg {segment_id}] {text}")
        segment_id += 1

    print(f"⏱️  耗时: {time.time() - start:.2f}s")


def demo_benchmark(args):
    """性能压测"""
    iterations = args.iterations or 5
    print(f"\n🔹 性能测试: {args.audio} × {iterations} 次")

    audio_data, sr = sf.read(args.audio)
    audio_duration = len(audio_data) / sr
    model = get_model()

    rtfs = []
    for i in range(iterations):
        start = time.time()
        model._model.generate(args.audio, language=args.language)
        elapsed = time.time() - start
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        rtfs.append(rtf)
        print(f"  Run {i + 1}: RTF={rtf:.4f}, Time={elapsed:.2f}s")

    avg_rtf = sum(rtfs) / len(rtfs)
    print(f"\n📊 平均 RTF: {avg_rtf:.4f} | 实时倍率: {1 / avg_rtf:.1f}x")


DEMOS = {
    "transcribe": demo_transcribe,
    "vad":        demo_vad,
    "stream":     demo_stream,
    "benchmark":  demo_benchmark,
    "all":        None,
}


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR Apple Silicon Demo")
    parser.add_argument("mode", choices=DEMOS.keys(), default="transcribe", nargs="?",
                        help="演示模式 (默认: transcribe)")
    parser.add_argument("audio", help="音频文件路径")
    parser.add_argument("--language", "-l", type=str, default="zh", help="语言代码 (zh, en, ja, ko)")
    parser.add_argument("--iterations", "-n", type=int, default=5, help="benchmark 迭代次数")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"❌ 文件不存在: {args.audio}")
        sys.exit(1)

    start = time.time()

    if args.mode == "all":
        for name, fn in DEMOS.items():
            if fn:
                fn(args)
    else:
        DEMOS[args.mode](args)

    print(f"\n🎉 完成！总耗时: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
