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


def _model_dir_from_name(model_name: str) -> Path:
    """Resolve default model directory from sherpa-onnx model name."""
    normalized = model_name.lower()

    if "sherpa-onnx-qwen3-asr-1.7b-int8" in normalized:
        return _default_models_root() / "sherpa-onnx-qwen3-asr-1.7B-int8"
    return _default_models_root() / "sherpa-onnx-qwen3-asr-0.6B-int8"


def _env_bool(name: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


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
        "--model-dir",
        type=Path,
        default=None,
        help="Path to model directory. If not set, it is auto-selected from model-name.",
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

    model_dir_env = os.environ.get("MODEL_DIR")
    if args.model_dir is not None:
        model_dir = args.model_dir.resolve()
    elif model_dir_env:
        model_dir = Path(model_dir_env).resolve()
    else:
        model_dir = _model_dir_from_name(args.model_name).resolve()

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
    )
