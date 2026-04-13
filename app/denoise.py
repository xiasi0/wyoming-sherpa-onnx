from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import sherpa_onnx

LOGGER = logging.getLogger("wyoming-sherpa-onnx")


class GtcrnEnhancer:
    def __init__(self, model_path: Path, num_threads: int = 1) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"GTCRN model file not found: {model_path}")

        config = sherpa_onnx.OnlineSpeechDenoiserConfig(
            model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(
                    model=str(model_path)
                ),
                debug=False,
                num_threads=max(1, int(num_threads)),
                provider="cpu",
            )
        )
        if not config.validate():
            raise RuntimeError("Invalid GTCRN denoiser config")

        self.denoiser = sherpa_onnx.OnlineSpeechDenoiser(config)
        self.sample_rate = int(self.denoiser.sample_rate)
        self.frame_shift = int(self.denoiser.frame_shift_in_samples)

        LOGGER.info(
            "GTCRN enhancer enabled: model=%s sample_rate=%d frame_shift=%d",
            model_path,
            self.sample_rate,
            self.frame_shift,
        )

    def reset(self) -> None:
        self.denoiser.reset()

    def enhance(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if waveform.size == 0:
            return waveform
        if int(sample_rate) != self.sample_rate:
            raise ValueError(
                f"GTCRN sample rate mismatch: got {sample_rate}, expected {self.sample_rate}"
            )

        output_chunks: list[np.ndarray] = []
        for start in range(0, waveform.size, self.frame_shift):
            chunk = np.ascontiguousarray(
                waveform[start : start + self.frame_shift], dtype=np.float32
            )
            denoised = self.denoiser(chunk, self.sample_rate)
            samples = np.asarray(denoised.samples, dtype=np.float32)
            if samples.size > 0:
                output_chunks.append(samples)

        if not output_chunks:
            return np.empty((0,), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(output_chunks))

    def flush(self) -> np.ndarray:
        tail = self.denoiser.flush()
        samples = np.asarray(tail.samples, dtype=np.float32)
        return np.ascontiguousarray(samples)
