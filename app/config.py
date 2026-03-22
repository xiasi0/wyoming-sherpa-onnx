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


def _env_path(name: str, default: str) -> Path:
    """Get path from environment variable."""
    return Path(os.environ.get(name, default))


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


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Wyoming ASR server backed by sherpa-onnx FunASR Nano."
    )
    parser.add_argument(
        "--host",
        default=_env_str("WYOMING_HOST", "0.0.0.0"),
        help="Server listen host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_env_int("WYOMING_PORT", 10300),
        help="Server listen port.",
    )
    parser.add_argument(
        "--service-name",
        default=_env_str("WYOMING_SERVICE_NAME", "sherpa-funasr"),
        help="mDNS service instance name for HA auto discovery.",
    )
    parser.add_argument(
        "--no-zeroconf",
        action="store_true",
        help="Disable mDNS service broadcast.",
    )
    parser.add_argument(
        "--model-name",
        default=_env_str("WYOMING_MODEL_NAME", "sherpa-onnx-funasr-nano-int8-2025-12-30"),
        help="Model name exposed in Wyoming info.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=_env_path("WYOMING_MODEL_DIR", "/data/models/sherpa-onnx-funasr-nano-int8-2025-12-30"),
        help="Path to model directory.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=_env_int("WYOMING_SAMPLE_RATE", 16000),
        help="ASR target sample rate.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=_env_int("WYOMING_NUM_THREADS", 2),
        help="Threads used by sherpa-onnx.",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        default=True,
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
    args = parser.parse_args()

    # --no-auto-download 覆盖 --auto-download
    auto_download = args.auto_download and not args.no_auto_download

    return AppConfig(
        host=args.host,
        port=args.port,
        service_name=args.service_name,
        enable_zeroconf=not args.no_zeroconf,
        model_name=args.model_name,
        model_dir=args.model_dir.resolve(),
        sample_rate=args.sample_rate,
        num_threads=args.num_threads,
        auto_download=auto_download,
        debug=args.debug,
    )
