from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sherpa_onnx

LOGGER = logging.getLogger("wyoming-sherpa-onnx")


@dataclass(slots=True)
class AudioFormat:
    rate: int
    width: int
    channels: int


@dataclass(slots=True)
class Qwen3AsrStream:
    pcm_buffer: bytearray = field(default_factory=bytearray)


class Qwen3AsrEngine:
    def __init__(self, model_dir: Path, sample_rate: int, num_threads: int) -> None:
        self.model_dir = model_dir
        self.sample_rate = sample_rate
        self.num_threads = num_threads
        self.recognizer = self._build_recognizer()
        LOGGER.debug(
            "Initialized Qwen3-ASR engine: model_dir=%s sample_rate=%s num_threads=%s",
            self.model_dir,
            self.sample_rate,
            self.num_threads,
        )

        self._width_divisors = {
            2: 32768.0,
            4: 2147483648.0,
            1: 128.0,
        }
        self._dtype_map = {2: np.int16, 4: np.int32, 1: np.uint8}

    def _build_recognizer(self) -> sherpa_onnx.OfflineRecognizer:
        factory = getattr(sherpa_onnx.OfflineRecognizer, "from_qwen3_asr", None)
        if not callable(factory):
            raise RuntimeError(
                "Current sherpa-onnx build does not expose OfflineRecognizer.from_qwen3_asr."
            )

        kwargs = {
            "conv_frontend": str(self.model_dir / "conv_frontend.onnx"),
            "encoder": str(self.model_dir / "encoder.int8.onnx"),
            "decoder": str(self.model_dir / "decoder.int8.onnx"),
            "tokenizer": str(self.model_dir / "tokenizer"),
            "num_threads": self.num_threads,
            "sample_rate": self.sample_rate,
            "debug": False,
        }
        LOGGER.debug("Qwen3-ASR factory kwargs resolved: %s", sorted(kwargs.keys()))
        return factory(**kwargs)

    def create_stream(self) -> Qwen3AsrStream:
        LOGGER.debug("Created offline ASR stream")
        return Qwen3AsrStream()

    def feed_audio_to_stream(
        self,
        stream: Qwen3AsrStream,
        pcm_bytes: bytes,
        fmt: AudioFormat,
    ) -> None:
        waveform = self._pcm_to_float32(pcm_bytes, fmt.width, fmt.channels)
        if fmt.rate != self.sample_rate:
            waveform = self._resample_linear(waveform, fmt.rate, self.sample_rate)
        if waveform.size == 0:
            return

        stream.pcm_buffer.extend((np.clip(waveform, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes())
        LOGGER.debug(
            "Accepted audio chunk: input_bytes=%s waveform_samples=%s buffered_pcm=%s",
            len(pcm_bytes),
            waveform.size,
            len(stream.pcm_buffer),
        )

    def finish_stream(self, stream: Qwen3AsrStream) -> str:
        if not stream.pcm_buffer:
            return ""

        recognizer_stream = self.recognizer.create_stream()
        waveform = np.frombuffer(bytes(stream.pcm_buffer), dtype=np.int16).astype(np.float32) / 32768.0
        if waveform.size == 0:
            return ""

        recognizer_stream.accept_waveform(self.sample_rate, waveform)
        self.recognizer.decode_stream(recognizer_stream)
        final_text = self._extract_text(recognizer_stream).strip()
        LOGGER.debug("Final transcript text: %r", final_text)
        return final_text

    def _extract_text(self, stream) -> str:
        stream_result = getattr(stream, "result", None)
        if stream_result is not None:
            text = getattr(stream_result, "text", "")
            if text:
                return str(text)

        get_result = getattr(self.recognizer, "get_result", None)
        if callable(get_result):
            result = get_result(stream)
            if isinstance(result, str):
                return result
            if isinstance(result, dict):
                return str(result.get("text", ""))

        return ""

    def _pcm_to_float32(self, pcm: bytes, width: int, channels: int) -> np.ndarray:
        dtype = self._dtype_map.get(width)
        if dtype is None:
            raise ValueError(f"Unsupported sample width: {width}")

        divisor = self._width_divisors[width]
        data = np.frombuffer(pcm, dtype=dtype).astype(np.float32) / divisor
        if width == 1:
            data -= 128.0 / divisor

        if channels > 1:
            usable = (data.size // channels) * channels
            if usable > 0:
                data = data[:usable].reshape(-1, channels).mean(axis=1)
        return data

    def _resample_linear(self, samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate or samples.size == 0:
            return samples

        duration = samples.size / float(src_rate)
        new_size = int(round(duration * dst_rate))
        if new_size <= 1:
            return np.empty(0, dtype=np.float32)

        src_x = np.linspace(0.0, duration, num=samples.size, endpoint=False, dtype=np.float64)
        dst_x = np.linspace(0.0, duration, num=new_size, endpoint=False, dtype=np.float64)
        return np.interp(dst_x, src_x, samples).astype(np.float32)
