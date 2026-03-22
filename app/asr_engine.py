from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sherpa_onnx


@dataclass(slots=True)
class AudioFormat:
    rate: int
    width: int
    channels: int


class FunAsrNanoEngine:
    def __init__(self, model_dir: Path, sample_rate: int, num_threads: int) -> None:
        self.model_dir = model_dir
        self.sample_rate = sample_rate
        self.num_threads = num_threads
        self.recognizer = self._build_recognizer()
        
        # 预计算音频转换参数，避免重复计算
        self._width_divisors = {
            2: 32768.0,
            4: 2147483648.0,
            1: 128.0,
        }

    def _build_recognizer(self):
        fn = sherpa_onnx.OfflineRecognizer.from_funasr_nano
        sig = inspect.signature(fn)

        candidates: dict[str, Any] = {
            "model": str(self.model_dir / "encoder_adaptor.int8.onnx"),
            "encoder_adaptor": str(self.model_dir / "encoder_adaptor.int8.onnx"),
            "encoder": str(self.model_dir / "encoder_adaptor.int8.onnx"),
            "llm": str(self.model_dir / "llm.int8.onnx"),
            "embedding": str(self.model_dir / "embedding.int8.onnx"),
            "tokenizer": str(self.model_dir / "Qwen3-0.6B"),
            "tokenizer_json": str(self.model_dir / "Qwen3-0.6B" / "tokenizer.json"),
            "tokens": str(self.model_dir / "Qwen3-0.6B" / "tokenizer.json"),
            "num_threads": self.num_threads,
            "sample_rate": self.sample_rate,
            "debug": False,
        }

        kwargs = {k: v for k, v in candidates.items() if k in sig.parameters}
        if not kwargs:
            raise RuntimeError("Unable to build FunASR Nano recognizer, unexpected API.")

        return fn(**kwargs)

    def create_stream(self) -> sherpa_onnx.OfflineStream:
        """创建一个新的识别流。

        Returns:
            OfflineStream 对象
        """
        return self.recognizer.create_stream()

    def feed_audio_to_stream(
        self,
        stream: sherpa_onnx.OfflineStream,
        pcm_bytes: bytes,
        fmt: AudioFormat,
    ) -> None:
        """向流中馈送音频数据。

        Args:
            stream: OfflineStream 对象
            pcm_bytes: PCM 音频数据
            fmt: 音频格式
        """
        waveform = self._pcm_to_float32(pcm_bytes, fmt.width, fmt.channels)
        if fmt.rate != self.sample_rate:
            waveform = self._resample_linear(waveform, fmt.rate, self.sample_rate)
        if waveform.size > 0:
            stream.accept_waveform(self.sample_rate, waveform)

    def finish_stream(self, stream: sherpa_onnx.OfflineStream) -> str:
        """完成流识别并返回结果。

        Args:
            stream: OfflineStream 对象

        Returns:
            识别结果文本
        """
        self.recognizer.decode_stream(stream)

        # 优化：直接访问 result 属性，避免多次 hasattr 检查
        stream_result = getattr(stream, "result", None)
        if stream_result is not None:
            text = getattr(stream_result, "text", "")
            if text:
                return str(text).strip()
        
        get_result = getattr(self.recognizer, "get_result", None)
        if get_result is not None:
            result = get_result(stream)
            if isinstance(result, str):
                return result.strip()
            if isinstance(result, dict):
                return str(result.get("text", "")).strip()
        return ""

    def transcribe_pcm(self, pcm_bytes: bytes, fmt: AudioFormat) -> str:
        """一次性识别 PCM 音频（向后兼容）。

        Args:
            pcm_bytes: PCM 音频数据
            fmt: 音频格式

        Returns:
            识别结果文本
        """
        if not pcm_bytes:
            return ""

        stream = self.create_stream()
        self.feed_audio_to_stream(stream, pcm_bytes, fmt)
        return self.finish_stream(stream)

    def _pcm_to_float32(self, pcm: bytes, width: int, channels: int) -> np.ndarray:
        """将 PCM 数据转换为 float32 波形。

        Args:
            pcm: PCM 字节数据
            width: 采样位深（字节）
            channels: 声道数

        Returns:
            float32 波形数组
        """
        # 使用预计算的除数
        divisor = self._width_divisors.get(width)
        if divisor is None:
            raise ValueError(f"Unsupported sample width: {width}")
        
        # 根据位深选择 dtype
        if width == 2:
            data = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / divisor
        elif width == 4:
            data = np.frombuffer(pcm, dtype=np.int32).astype(np.float32) / divisor
        elif width == 1:
            data = (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0) / divisor
        else:
            raise ValueError(f"Unsupported sample width: {width}")

        # 多声道混合为单声道
        if channels > 1:
            usable = (data.size // channels) * channels
            if usable > 0:
                data = data[:usable].reshape(-1, channels).mean(axis=1)
        return data

    def _resample_linear(self, samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """线性重采样音频。

        Args:
            samples: 输入样本
            src_rate: 源采样率
            dst_rate: 目标采样率

        Returns:
            重采样后的样本
        """
        if src_rate == dst_rate or samples.size == 0:
            return samples

        duration = samples.size / float(src_rate)
        new_size = int(round(duration * dst_rate))
        if new_size <= 1:
            return np.array([], dtype=np.float32)

        # 优化：避免重复创建数组
        src_x = np.linspace(0.0, duration, num=samples.size, endpoint=False)
        dst_x = np.linspace(0.0, duration, num=new_size, endpoint=False)
        return np.interp(dst_x, src_x, samples).astype(np.float32)
