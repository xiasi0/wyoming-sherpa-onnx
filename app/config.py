from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    """Get integer from environment variable."""
    value = os.environ.get(name)
    return int(value) if value else default


def _env_str(name: str, default: str) -> str:
    """Get string from environment variable."""
    return os.environ.get(name, default)


def _default_models_root() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "models"


def _default_speaker_refs_root() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "speaker_refs"


def _model_dir_from_name(model_name: str) -> Path:
    """Resolve default model directory from sherpa-onnx model name."""
    normalized = model_name.lower()

    if "sherpa-onnx-qwen3-asr-1.7b-int8" in normalized:
        return _default_models_root() / "sherpa-onnx-qwen3-asr-1.7B-int8"
    return _default_models_root() / "sherpa-onnx-qwen3-asr-0.6B-int8"


def _speaker_model_dir_from_name(model_name: str) -> Path:
    _ = model_name
    return _default_models_root() / "speaker"


def _denoise_model_dir_from_name(model_name: str) -> Path:
    _ = model_name
    return _default_models_root() / "denoise"


_SPEAKER_MODEL_NAME_TO_FILE = {
    "wespeaker-zh-cnceleb-resnet34": "wespeaker_zh_cnceleb_resnet34.onnx",
    "wespeaker_zh_cnceleb_resnet34": "wespeaker_zh_cnceleb_resnet34.onnx",
    "wespeaker-en-voxceleb-resnet34": "wespeaker_en_voxceleb_resnet34.onnx",
    "wespeaker_en_voxceleb_resnet34": "wespeaker_en_voxceleb_resnet34.onnx",
    "3dspeaker-campplus-zh-en": "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
    "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced": "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
}

_DENOISE_MODEL_NAME_TO_FILE = {
    "gtcrn-simple": "gtcrn_simple.onnx",
    "gtcrn_simple": "gtcrn_simple.onnx",
}

_SPEAKER_MODEL_BASE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models"
)

_DENOISE_MODEL_BASE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speech-enhancement-models"
)


def _env_bool(name: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def _is_under_speaker_subdir(wav_path: Path, refs_root: Path) -> bool:
    try:
        rel = wav_path.resolve().relative_to(refs_root.resolve())
    except ValueError:
        return False
    # Require at least: <speaker_id>/<file.wav>
    return len(rel.parts) >= 2


@dataclass(slots=True)
class AppConfig:
    host: str
    port: int
    service_name: str
    enable_zeroconf: bool
    model_name: str
    model_dir: Path
    sample_rate: int
    num_threads: int
    auto_download: bool
    debug: bool
    hotwords: str
    speaker_gate: bool
    speaker_threshold: float
    speaker_model_dir: Path
    speaker_model_file: str
    speaker_model_url: str
    speaker_reference_dir: Path
    speaker_reference_wavs: tuple[Path, ...]
    denoise_enabled: bool
    denoise_model_dir: Path
    denoise_model_file: str
    denoise_model_url: str


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Wyoming ASR server for local Qwen3-ASR models."
    )
    parser.add_argument(
        "--host",
        default=_env_str("HOST", "0.0.0.0"),
        help="Server listen host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_env_int("PORT", 10300),
        help="Server listen port.",
    )
    parser.add_argument(
        "--service-name",
        default=_env_str("SERVICE_NAME", "wyoming-sherpa-onnx"),
        help="mDNS service instance name for HA auto discovery.",
    )
    parser.add_argument(
        "--zeroconf",
        action="store_true",
        default=None,
        help="Enable mDNS service broadcast.",
    )
    parser.add_argument(
        "--no-zeroconf",
        action="store_true",
        help="Disable mDNS service broadcast.",
    )
    parser.add_argument(
        "--model-name",
        default=_env_str("MODEL_NAME", "sherpa-onnx-qwen3-asr-0.6B-int8"),
        help="Model name exposed in Wyoming info.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=_env_int("SAMPLE_RATE", 16000),
        help="ASR target sample rate.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=_env_int("NUM_THREADS", 2),
        help="Threads used by sherpa-onnx.",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        default=None,
        help="Auto download model if not exists (default: True).",
    )
    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Disable auto download of model files.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--hotwords",
        default=_env_str("HOTWORDS", ""),
        help="Global hotwords (comma-separated).",
    )
    parser.add_argument(
        "--speaker-gate",
        action="store_true",
        default=None,
        help="Enable speaker gate. Non-target speakers are rejected.",
    )
    parser.add_argument(
        "--no-speaker-gate",
        action="store_true",
        help="Disable speaker gate.",
    )
    parser.add_argument(
        "--speaker-threshold",
        type=float,
        default=float(_env_str("SPEAKER_THRESHOLD", "0.40")),
        help="Cosine similarity threshold for speaker gate.",
    )
    parser.add_argument(
        "--speaker-model-dir",
        type=Path,
        default=None,
        help="Directory for speaker embedding model.",
    )
    parser.add_argument(
        "--speaker-model-name",
        default=_env_str("SPEAKER_MODEL_NAME", "wespeaker-zh-cnceleb-resnet34"),
        help="Speaker embedding model name alias.",
    )
    parser.add_argument(
        "--speaker-model-file",
        default=_env_str("SPEAKER_MODEL_FILE", ""),
        help="Speaker embedding model filename override.",
    )
    parser.add_argument(
        "--speaker-model-url",
        default=_env_str("SPEAKER_MODEL_URL", ""),
        help="Direct download URL of speaker embedding ONNX model.",
    )
    parser.add_argument(
        "--speaker-reference-wavs",
        default=_env_str("SPEAKER_REFERENCE_WAVS", ""),
        help="Comma-separated enrollment wav files for target speaker. "
        "If omitted, auto-load *.wav from speaker-reference-dir recursively.",
    )
    parser.add_argument(
        "--speaker-reference-dir",
        type=Path,
        default=Path(_env_str("SPEAKER_REFERENCE_DIR", str(_default_speaker_refs_root()))),
        help="Root directory for speaker enrollment wavs (supports sub-directories).",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        default=None,
        help="Enable GTCRN denoise frontend before speaker-gate/ASR.",
    )
    parser.add_argument(
        "--denoise-model-name",
        default=_env_str("DENOISE_MODEL_NAME", "gtcrn-simple"),
        help="Denoise model name alias.",
    )
    parser.add_argument(
        "--denoise-model-file",
        default=_env_str("DENOISE_MODEL_FILE", ""),
        help="Denoise model filename override.",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable GTCRN denoise frontend.",
    )
    args = parser.parse_args()

    # --no-zeroconf 覆盖 --zeroconf
    if args.zeroconf is None:
        # 从环境变量读取
        enable_zeroconf = _env_bool("ZEROCONF", False)
    else:
        enable_zeroconf = args.zeroconf
    if args.no_zeroconf:
        enable_zeroconf = False

    # --no-auto-download 覆盖 --auto-download
    if args.auto_download is None:
        auto_download = _env_bool("AUTO_DOWNLOAD", True)
    else:
        auto_download = args.auto_download
    auto_download = auto_download and not args.no_auto_download

    if args.speaker_gate is None:
        speaker_gate = _env_bool("SPEAKER_GATE", False)
    else:
        speaker_gate = args.speaker_gate
    if args.no_speaker_gate:
        speaker_gate = False

    if args.denoise is None:
        denoise_enabled = _env_bool("DENOISE", True)
    else:
        denoise_enabled = args.denoise
    if args.no_denoise:
        denoise_enabled = False

    model_dir = _model_dir_from_name(args.model_name).resolve()

    speaker_reference_value = str(args.speaker_reference_wavs or "").strip()
    if speaker_reference_value:
        refs_root = args.speaker_reference_dir.expanduser().resolve()
        speaker_reference_wavs = tuple(
            p
            for p in (
                Path(item.strip()).expanduser().resolve()
                for item in speaker_reference_value.split(",")
                if item.strip()
            )
            if _is_under_speaker_subdir(p, refs_root)
        )
    else:
        refs_root = args.speaker_reference_dir.expanduser().resolve()
        speaker_reference_wavs = tuple(
            sorted(p for p in refs_root.glob("**/*.wav") if _is_under_speaker_subdir(p, refs_root))
        )

    if args.speaker_model_dir is None:
        speaker_model_dir = Path(
            _env_str("SPEAKER_MODEL_DIR", str(_speaker_model_dir_from_name(args.model_name)))
        ).expanduser().resolve()
    else:
        speaker_model_dir = args.speaker_model_dir.expanduser().resolve()

    speaker_model_name = str(args.speaker_model_name or "").strip().lower()
    speaker_model_file = str(args.speaker_model_file or "").strip()
    if not speaker_model_file:
        speaker_model_file = _SPEAKER_MODEL_NAME_TO_FILE.get(speaker_model_name, "")
    if not speaker_model_file:
        parser.error(
            "Unknown speaker model name. Use --speaker-model-name "
            "'wespeaker-zh-cnceleb-resnet34' / 'wespeaker-en-voxceleb-resnet34' / "
            "'3dspeaker-campplus-zh-en', or provide --speaker-model-file."
        )
    speaker_model_url = str(args.speaker_model_url or "").strip()
    if not speaker_model_url:
        speaker_model_url = f"{_SPEAKER_MODEL_BASE_URL}/{speaker_model_file}"

    denoise_model_dir = Path(
        _env_str("DENOISE_MODEL_DIR", str(_denoise_model_dir_from_name(args.model_name)))
    ).expanduser().resolve()

    denoise_model_name = str(args.denoise_model_name or "").strip().lower()
    denoise_model_file = str(args.denoise_model_file or "").strip()
    if not denoise_model_file:
        denoise_model_file = _DENOISE_MODEL_NAME_TO_FILE.get(denoise_model_name, "")
    if not denoise_model_file:
        parser.error(
            "Unknown denoise model name. Use --denoise-model-name "
            "'gtcrn-simple' or provide --denoise-model-file."
        )
    denoise_model_url = _env_str("DENOISE_MODEL_URL", "").strip()
    if not denoise_model_url:
        denoise_model_url = f"{_DENOISE_MODEL_BASE_URL}/{denoise_model_file}"

    return AppConfig(
        host=args.host,
        port=args.port,
        service_name=args.service_name,
        enable_zeroconf=enable_zeroconf,
        model_name=args.model_name,
        model_dir=model_dir,
        sample_rate=args.sample_rate,
        num_threads=args.num_threads,
        auto_download=auto_download,
        debug=args.debug,
        hotwords=str(args.hotwords or "").strip(),
        speaker_gate=speaker_gate,
        speaker_threshold=float(args.speaker_threshold),
        speaker_model_dir=speaker_model_dir,
        speaker_model_file=speaker_model_file,
        speaker_model_url=speaker_model_url,
        speaker_reference_dir=args.speaker_reference_dir.expanduser().resolve(),
        speaker_reference_wavs=speaker_reference_wavs,
        denoise_enabled=denoise_enabled,
        denoise_model_dir=denoise_model_dir,
        denoise_model_file=denoise_model_file,
        denoise_model_url=denoise_model_url,
    )
