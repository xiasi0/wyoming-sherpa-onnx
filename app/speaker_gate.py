from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

LOGGER = logging.getLogger("wyoming-sherpa-onnx")


def _read_wav_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported wav sample width: {sample_width} bytes ({path})")

    if channels > 1:
        usable = (data.size // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1)

    return np.ascontiguousarray(data, dtype=np.float32), sample_rate


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


class SpeakerGate:
    def __init__(
        self,
        model_path: Path,
        reference_wavs: tuple[Path, ...],
        threshold: float = 0.60,
        num_threads: int = 1,
        reference_root: Path | None = None,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Speaker model file not found: {model_path}")
        if model_path.suffix.lower() != ".onnx":
            raise ValueError(f"Speaker model must be an ONNX file: {model_path}")
        if not reference_wavs:
            raise ValueError("speaker gate enabled but no speaker reference wavs provided")

        self.model_path = model_path
        self.threshold = threshold
        self.reference_root = reference_root.resolve() if reference_root else None

        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model_path),
            num_threads=max(1, int(num_threads)),
            debug=False,
            provider="cpu",
        )
        if not cfg.validate():
            raise RuntimeError(f"Invalid speaker embedding config: {cfg}")

        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.speaker_embeddings = self._build_speaker_embeddings(reference_wavs)

        LOGGER.info(
            "Speaker gate enabled: model=%s references=%d speakers=%d threshold=%.3f dim=%d",
            model_path,
            len(reference_wavs),
            len(self.speaker_embeddings),
            self.threshold,
            self.extractor.dim,
        )

    def _compute_embedding(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=waveform)
        stream.input_finished()
        if not self.extractor.is_ready(stream):
            return np.zeros(self.extractor.dim, dtype=np.float32)
        embedding = np.array(self.extractor.compute(stream), dtype=np.float32)
        return np.ascontiguousarray(embedding)

    def _speaker_id_from_path(self, wav_path: Path) -> str:
        if self.reference_root is not None:
            try:
                rel = wav_path.resolve().relative_to(self.reference_root)
                if len(rel.parts) >= 2:
                    return rel.parts[0]
                return "default"
            except ValueError:
                pass
        return wav_path.parent.name or "default"

    def _build_speaker_embeddings(
        self, reference_wavs: tuple[Path, ...]
    ) -> dict[str, np.ndarray]:
        grouped: dict[str, list[np.ndarray]] = {}
        for wav_path in reference_wavs:
            if not wav_path.exists():
                raise FileNotFoundError(f"Speaker reference wav not found: {wav_path}")
            waveform, sample_rate = _read_wav_mono_float32(wav_path)
            if waveform.size == 0:
                continue
            speaker_id = self._speaker_id_from_path(wav_path)
            grouped.setdefault(speaker_id, []).append(
                self._compute_embedding(waveform, sample_rate)
            )

        if not grouped:
            raise RuntimeError("No valid speaker reference wavs could produce embeddings")

        merged: dict[str, np.ndarray] = {}
        for speaker_id, embeddings in grouped.items():
            merged_vec = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
            merged[speaker_id] = np.ascontiguousarray(merged_vec)
            LOGGER.info(
                "Speaker profile loaded: id=%s refs=%d",
                speaker_id,
                len(embeddings),
            )
        return merged

    def accepts_waveform(
        self, waveform: np.ndarray, sample_rate: int, threshold: float | None = None
    ) -> tuple[bool, float, str | None]:
        if waveform.size == 0:
            return False, 0.0, None
        embedding = self._compute_embedding(waveform, sample_rate)
        best_speaker: str | None = None
        best_similarity = -1.0
        for speaker_id, target_embedding in self.speaker_embeddings.items():
            similarity = _cosine_similarity(embedding, target_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker = speaker_id
        use_threshold = self.threshold if threshold is None else float(threshold)
        return best_similarity >= use_threshold, best_similarity, best_speaker
